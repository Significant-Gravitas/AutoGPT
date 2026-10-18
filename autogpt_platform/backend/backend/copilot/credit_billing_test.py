"""AutoPilot block turns billed to credits once the plan's allowance is out.

The executor runs a block turn on the credit balance instead of refusing it
when the balance covers the per-turn spend floor. The turn is capped at what
the balance covers and charged the cost it recorded when it ends.
"""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest
from claude_agent_sdk import AssistantMessage, TextBlock

from backend.copilot.config import ChatConfig
from backend.copilot.constants import AUTOPILOT_BLOCK_ID
from backend.copilot.credit_billing import (
    CreditBilledTurn,
    bill_to_credits,
    credit_billing,
    record_credit_billed_cost,
    wait_for_cost,
)
from backend.copilot.rate_limit import RateLimitUnavailable
from backend.copilot.route_table import PlatformRoute, platform_row
from backend.copilot.sdk.governor import TurnGovernor
from backend.copilot.sdk.usage_meter import UsageMeter, cli_price_ratio
from backend.copilot.turn_pause import TurnPause
from backend.util.exceptions import InsufficientBalanceError

OUT_OF_TODAY = TurnPause(
    reason="daily",
    resets_at=datetime(2026, 10, 15, tzinfo=UTC),
    checkpoint_saved=False,
    started=False,
)
KIMI = "moonshotai/kimi-k2.6"

_CREDITS_ON = "backend.copilot.credit_billing.settings.config.enable_credit"


@pytest.fixture
def credits(mocker) -> MagicMock:
    db = MagicMock()
    db.get_credits = AsyncMock(return_value=500)
    db.spend_credits = AsyncMock(return_value=0)
    db.get_org_credits = AsyncMock(return_value=0)
    db.spend_org_credits = AsyncMock(return_value=0)
    mocker.patch("backend.copilot.credit_billing.credit_db", return_value=db)
    # A deployment that bills credits, on a route whose engine caps a turn's
    # dollars.
    mocker.patch(_CREDITS_ON, True)
    mocker.patch.object(
        ChatConfig,
        "route_row",
        new_callable=PropertyMock,
        return_value=platform_row("openrouter"),
    )
    return db


async def _billed(credits: MagicMock, balance: int) -> CreditBilledTurn:
    credits.get_credits.return_value = balance
    turn = await bill_to_credits(
        user_id="user-1",
        session_id="sess-1",
        tool_name="autopilot_block",
        refusal=OUT_OF_TODAY,
    )
    assert turn is not None
    return turn


def _charged(credits: MagicMock) -> list[int]:
    return [call.kwargs["cost"] for call in credits.spend_credits.await_args_list]


class _Meter(UsageMeter):
    def __init__(self, spent: float, last: float) -> None:
        super().__init__()
        self.spent = spent
        self.last = last

    def observe(self, message: AssistantMessage) -> None:
        pass

    @property
    def spent_usd(self) -> float:
        return self.spent

    @property
    def last_main_call_usd(self) -> float:
        return self.last


@pytest.mark.asyncio
async def test_a_block_turn_runs_when_the_balance_covers_the_floor(credits):
    turn = await _billed(credits, balance=50)
    assert turn.cap_credits == 50

    credits.get_credits.return_value = 49
    assert (
        await bill_to_credits(
            user_id="user-1",
            session_id="sess-1",
            tool_name="autopilot_block",
            refusal=OUT_OF_TODAY,
        )
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["chat", "run_sub_session", None])
async def test_only_the_blocks_turns_bill_to_credits(credits, tool_name):
    turn = await bill_to_credits(
        user_id="user-1", session_id="sess-1", tool_name=tool_name, refusal=OUT_OF_TODAY
    )

    assert turn is None
    credits.get_credits.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_turn_bills_to_credits_where_credits_are_off(credits, mocker):
    # The disabled credit model reads a balance and charges nothing: a turn
    # billed there would run free.
    mocker.patch(_CREDITS_ON, False)

    assert await _bill(organization_id=None) is None
    credits.get_credits.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["local", None], ids=["local", "disabled"])
async def test_no_turn_bills_to_credits_without_the_engine_that_caps_it(
    credits, mocker, route: PlatformRoute | None
):
    """The local route's engine has no dollar ceiling, and a disabled route
    runs no turn of its own."""
    mocker.patch.object(
        ChatConfig,
        "route_row",
        new_callable=PropertyMock,
        return_value=platform_row(route) if route else None,
    )

    assert await _bill(organization_id=None) is None
    credits.get_credits.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_organizations_run_bills_the_organizations_balance(credits):
    credits.get_credits.return_value = 0
    credits.get_org_credits.return_value = 300

    turn = await _bill(organization_id="org-1", team_id="team-1")
    assert turn is not None and turn.cap_credits == 300
    await turn.record(1_000_000)
    await turn.end()

    credits.get_org_credits.assert_awaited_once_with(org_id="org-1")
    credits.spend_credits.assert_not_awaited()
    charge = credits.spend_org_credits.await_args.kwargs
    assert (charge["org_id"], charge["user_id"], charge["team_id"]) == (
        "org-1",
        "user-1",
        "team-1",
    )
    assert charge["cost"] == 100


@pytest.mark.asyncio
async def test_a_turn_cancelled_while_its_charge_waits_is_still_charged(credits):
    turn = await _billed(credits, balance=500)
    await turn.record(1_000_000)
    reconcile = asyncio.create_task(asyncio.sleep(3600))
    turn.wait_for(reconcile)

    ending = asyncio.create_task(turn.end())
    await asyncio.sleep(0)
    ending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await ending
    reconcile.cancel()

    assert _charged(credits) == [100]


async def _bill(**tenancy) -> CreditBilledTurn | None:
    return await bill_to_credits(
        user_id="user-1",
        session_id="sess-1",
        tool_name="autopilot_block",
        refusal=OUT_OF_TODAY,
        **tenancy,
    )


@pytest.mark.asyncio
async def test_an_unreadable_balance_fails_closed(credits):
    credits.get_credits.side_effect = ConnectionError("db down")

    with pytest.raises(RateLimitUnavailable):
        await bill_to_credits(
            user_id="user-1",
            session_id="sess-1",
            tool_name="autopilot_block",
            refusal=OUT_OF_TODAY,
        )


@pytest.mark.asyncio
async def test_the_budget_is_the_balance_within_the_per_turn_ceiling(credits):
    budget = (await _billed(credits, balance=75)).budget
    assert (budget.usd, budget.hard_cap) == (0.75, True)
    assert (budget.limiting, budget.resets_at) == ("daily", OUT_OF_TODAY.resets_at)

    assert (await _billed(credits, balance=5_000)).budget.usd == 10.0


@pytest.mark.asyncio
async def test_a_turn_costing_more_than_the_balance_stops_at_the_cap(credits):
    turn = await _billed(credits, balance=75)
    meter = _Meter(spent=0.70, last=0.01)
    governor = TurnGovernor(budget=turn.budget, step_checkin=450)
    governor.meter = meter

    # The CLI's own stop sits on the balance, with none of a paid plan's
    # headroom past it.
    assert governor.cli_budget_usd(KIMI) == pytest.approx(0.75 * cli_price_ratio(KIMI))
    governor.observe(AssistantMessage(content=[TextBlock(text="x")], model="m"))
    assert governor.landing is None
    meter.spent = 0.74
    governor.observe(AssistantMessage(content=[TextBlock(text="x")], model="m"))
    assert governor.landing == "daily"

    # Spend the CLI let past the cap before it stopped is never charged.
    async with credit_billing(turn):
        assert await record_credit_billed_cost("user-1", 900_000)
    assert _charged(credits) == [75]


@pytest.mark.asyncio
async def test_the_turn_is_charged_its_recorded_cost_when_it_ends(credits):
    turn = await _billed(credits, balance=500)

    async with credit_billing(turn):
        assert await record_credit_billed_cost("user-1", 123_400)
        assert await record_credit_billed_cost("user-1", 1)
        credits.spend_credits.assert_not_awaited()

    # 123,401 microdollars, rounded up to whole credits, in one charge.
    credits.spend_credits.assert_awaited_once()
    charge = credits.spend_credits.await_args.kwargs
    assert (charge["user_id"], charge["cost"]) == ("user-1", 13)
    assert charge["metadata"].chat_session_id == "sess-1"
    assert charge["metadata"].block_id == AUTOPILOT_BLOCK_ID
    assert charge["metadata"].block == "AutoPilotBlock"


@pytest.mark.asyncio
async def test_the_charge_waits_for_cost_recorded_after_the_stream(credits):
    turn = await _billed(credits, balance=500)

    async def reconcile() -> None:
        await asyncio.sleep(0.05)
        await record_credit_billed_cost("user-1", 50_000)

    async with credit_billing(turn):
        wait_for_cost(asyncio.create_task(reconcile()))

    assert _charged(credits) == [5]


@pytest.mark.asyncio
async def test_cost_recorded_after_the_charge_is_charged_when_it_lands(credits):
    turn = await _billed(credits, balance=500)
    async with credit_billing(turn):
        await record_credit_billed_cost("user-1", 25_000)

    await turn.record(30_000)

    # 55,000 microdollars in all: 6 credits, not 3 + 3.
    assert _charged(credits) == [3, 3]
    assert turn.charged_credits == 6


@pytest.mark.asyncio
async def test_a_balance_spent_meanwhile_is_charged_only_what_is_left(credits):
    turn = await _billed(credits, balance=500)
    credits.spend_credits.side_effect = [
        InsufficientBalanceError("too low", "user-1", 20, 40),
        0,
    ]

    async with credit_billing(turn):
        await record_credit_billed_cost("user-1", 400_000)
        credits.get_credits.return_value = 20

    assert _charged(credits) == [40, 20]
    assert turn.charged_credits == 20


@pytest.mark.asyncio
async def test_another_users_cost_stays_off_the_turn(credits):
    turn = await _billed(credits, balance=500)

    async with credit_billing(turn):
        assert not await record_credit_billed_cost("user-2", 90_000)

    credits.spend_credits.assert_not_awaited()
