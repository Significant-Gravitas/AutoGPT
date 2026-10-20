"""AutoPilot block turns billed to credits once the plan's allowance is out.

The executor runs a block turn on the credit balance instead of refusing it
when the balance covers the per-turn spend floor. The turn is capped at what
the balance covers and charged the cost it records as it lands: each whole
credit as the cost reaches it, and the rest, rounded up, when it ends.
"""

import asyncio
import logging
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
from backend.copilot.sdk.governor import TurnGovernor
from backend.copilot.sdk.usage_meter import UsageMeter, cli_price_ratio
from backend.copilot.turn_pause import TurnPause
from backend.data.credit import UsageTransactionMetadata
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
    # A deployment that bills credits, on a route the SDK engine can take.
    mocker.patch(_CREDITS_ON, True)
    mocker.patch.object(
        ChatConfig, "thinking_available", new_callable=PropertyMock, return_value=True
    )
    return db


class _Wallet:
    """The account's balance. A debit it can't cover is refused whole, as the
    credit model refuses it."""

    def __init__(self, balance: int) -> None:
        self.balance = balance
        self.debits: list[int] = []

    async def get_credits(self, user_id: str) -> int:
        return self.balance

    async def spend_credits(
        self, *, user_id: str, cost: int, metadata: UsageTransactionMetadata
    ) -> int:
        # A debit is a round trip: other charges run while it is in flight.
        await asyncio.sleep(0)
        if cost > self.balance:
            raise InsufficientBalanceError("too low", user_id, self.balance, cost)
        self.balance -= cost
        self.debits.append(cost)
        return self.balance


@pytest.fixture
def wallet(credits: MagicMock) -> _Wallet:
    """500 credits for the turn's debits to draw on."""
    wallet = _Wallet(balance=500)
    credits.get_credits.side_effect = wallet.get_credits
    credits.spend_credits.side_effect = wallet.spend_credits
    return wallet


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


def _leaks(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage().startswith("Billing leak")]


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
async def test_no_turn_bills_to_credits_without_the_engine_that_caps_it(
    credits, mocker
):
    mocker.patch.object(
        ChatConfig, "thinking_available", new_callable=PropertyMock, return_value=False
    )

    assert await _bill(organization_id=None) is None


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
    await turn.record(1_005_000)
    reconcile = asyncio.create_task(asyncio.sleep(3600))
    turn.wait_for(reconcile)

    ending = asyncio.create_task(turn.end())
    await asyncio.sleep(0)
    ending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await ending
    reconcile.cancel()

    # The 100 whole credits as the cost landed, and the half credit left.
    assert _charged(credits) == [100, 1]


async def _bill(**tenancy) -> CreditBilledTurn | None:
    return await bill_to_credits(
        user_id="user-1",
        session_id="sess-1",
        tool_name="autopilot_block",
        refusal=OUT_OF_TODAY,
        **tenancy,
    )


async def _turn() -> CreditBilledTurn:
    turn = await _bill()
    assert turn is not None
    return turn


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
async def test_the_turn_is_charged_whole_credits_as_its_cost_lands(credits):
    turn = await _billed(credits, balance=500)

    async with credit_billing(turn):
        assert await record_credit_billed_cost("user-1", 123_400)
        assert _charged(credits) == [12]
        assert await record_credit_billed_cost("user-1", 1)
        assert _charged(credits) == [12]

    # 123,401 microdollars: the 12 whole credits as they landed, and the part
    # credit left, rounded up, when the turn ended.
    assert _charged(credits) == [12, 1]
    for call in credits.spend_credits.await_args_list:
        assert call.kwargs["user_id"] == "user-1"
        assert call.kwargs["metadata"].chat_session_id == "sess-1"
        assert call.kwargs["metadata"].block_id == AUTOPILOT_BLOCK_ID
        assert call.kwargs["metadata"].block == "AutoPilotBlock"


@pytest.mark.asyncio
async def test_part_credits_are_charged_once_they_add_up_to_a_whole_one(credits):
    turn = await _billed(credits, balance=500)

    for _ in range(7):
        await turn.record(5_000)
    assert _charged(credits) == [1, 1, 1]

    await turn.end()
    # 3.5 credits in all, rounded up.
    assert _charged(credits) == [1, 1, 1, 1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "costs, balance",
    [
        ([1], 500),
        ([9_999, 1], 500),
        ([123_400, 1], 500),
        ([5_000] * 7, 500),
        ([33_333, 33_333, 33_334], 500),
        ([123_456, 250_000, 99_999], 500),
        # Past the cap: the balance the turn started with.
        ([250_001, 499_999, 3], 60),
        ([600_001], 60),
    ],
)
async def test_the_turn_pays_what_one_charge_at_its_end_would_have(
    credits, wallet, costs, balance
):
    wallet.balance = balance
    turn = await _turn()

    async with credit_billing(turn):
        for cost in costs:
            await record_credit_billed_cost("user-1", cost)
        # Every whole credit is in before the turn ends.
        assert sum(wallet.debits) == min(sum(costs) // 10_000, balance)

    # The recorded cost rounded up to whole credits, within the cap: what the
    # turn's one charge took when nothing was charged before its end.
    assert sum(wallet.debits) == turn.charged_credits
    assert turn.charged_credits == min(-(-sum(costs) // 10_000), balance)


@pytest.mark.asyncio
async def test_concurrent_records_never_charge_more_than_the_turn_spent(
    credits, wallet
):
    turn = await _turn()

    await asyncio.gather(*(turn.record(12_500) for _ in range(8)))
    # 100,000 microdollars: 10 credits, however the charges interleaved.
    assert sum(wallet.debits) == turn.charged_credits == 10

    await asyncio.gather(turn.record(12_500), turn.end(), turn.record(12_500))
    # 125,000 in all, with the end racing the last two: 12.5, rounded up once.
    assert sum(wallet.debits) == turn.charged_credits == 13


@pytest.mark.asyncio
async def test_a_turn_that_never_ends_has_paid_all_but_a_part_credit(credits, wallet):
    """A process that dies mid-turn never reaches the charge at the end."""
    turn = await _turn()

    for cost in (123_456, 250_000, 99_999):
        await turn.record(cost)

    # 473,455 microdollars recorded: 47 credits paid, 3,455 unpaid.
    assert wallet.debits == [12, 25, 10]
    assert 473_455 - 10_000 * sum(wallet.debits) < 10_000


@pytest.mark.asyncio
async def test_a_charge_still_counts_when_its_recorder_is_cancelled(credits):
    """A debit the ledger took can't be taken back: it counts, and no later
    charge takes it again."""
    taken = asyncio.Event()
    replied = asyncio.Event()

    async def debit_awaiting_its_reply(**charge) -> int:
        taken.set()
        await replied.wait()
        return 0

    turn = await _billed(credits, balance=500)
    credits.spend_credits.side_effect = debit_awaiting_its_reply
    recording = asyncio.create_task(turn.record(25_000))
    await asyncio.wait_for(taken.wait(), timeout=5)
    recording.cancel()
    with pytest.raises(asyncio.CancelledError):
        await recording
    replied.set()
    await turn.end()

    assert _charged(credits) == [2, 1]
    assert turn.charged_credits == 3


@pytest.mark.asyncio
async def test_a_failed_debit_counts_as_charged(credits, caplog):
    # It may have landed, and charging it again could take it twice.
    caplog.set_level(logging.WARNING, logger="backend.copilot.credit_billing")
    turn = await _billed(credits, balance=500)
    credits.spend_credits.side_effect = [ConnectionError("db blip"), 0, 0]

    await turn.record(25_000)
    assert turn.charged_credits == 2
    await turn.record(10_000)
    assert turn.charged_credits == 3
    await turn.end()

    assert _charged(credits) == [2, 1, 1]
    assert turn.charged_credits == 4
    assert [leak.levelno for leak in _leaks(caplog)] == [logging.ERROR]


@pytest.mark.asyncio
async def test_a_debit_that_lands_then_raises_is_not_taken_again(
    credits, wallet, caplog
):
    # The credit model commits a debit before it looks up the user's auto
    # top-up, which can still raise: the credits are gone, but the call failed.
    caplog.set_level(logging.WARNING, logger="backend.copilot.credit_billing")
    turn = await _billed(credits, balance=500)

    async def lands_then_raises(**debit) -> int:
        await wallet.spend_credits(**debit)
        raise RuntimeError("auto top-up settings unreadable")

    credits.spend_credits.side_effect = lands_then_raises
    await turn.record(25_000)
    credits.spend_credits.side_effect = wallet.spend_credits
    await turn.end()

    # 25,000 microdollars rounds up to 3 credits, taken once.
    assert wallet.debits == [2, 1]
    assert wallet.balance == 497
    assert [leak.levelno for leak in _leaks(caplog)] == [logging.ERROR]


@pytest.mark.asyncio
async def test_what_is_left_counts_as_charged_once_its_debit_may_have_landed(
    credits, wallet, caplog
):
    caplog.set_level(logging.WARNING, logger="backend.copilot.credit_billing")
    turn = await _billed(credits, balance=500)
    # Other runs spent all but a credit meanwhile.
    wallet.balance = 1

    async def lands_then_raises(**debit) -> int:
        await wallet.spend_credits(**debit)
        raise RuntimeError("reply lost")

    credits.spend_credits.side_effect = lands_then_raises
    await turn.record(25_000)
    # A top-up lands before the turn ends.
    wallet.balance += 100
    credits.spend_credits.side_effect = wallet.spend_credits
    await turn.end()

    # The credit that was left, then the 2 more the 3 owed come to.
    assert wallet.debits == [1, 2]
    assert turn.charged_credits == 3
    assert [leak.levelno for leak in _leaks(caplog)] == [logging.ERROR]


@pytest.mark.asyncio
async def test_a_balance_it_cannot_read_takes_nothing_and_stays_due(credits, wallet):
    turn = await _billed(credits, balance=500)
    # Other runs spent all but a credit, and the balance can't be read.
    wallet.balance = 1
    credits.get_credits.side_effect = ConnectionError("db down")

    await turn.record(25_000)
    assert (turn.charged_credits, wallet.debits) == (0, [])

    # Once it reads again, after a top-up, the end takes all that is due.
    credits.get_credits.side_effect = wallet.get_credits
    wallet.balance += 100
    await turn.end()
    assert (turn.charged_credits, wallet.debits) == (3, [3])


@pytest.mark.asyncio
async def test_what_is_left_that_goes_before_it_is_taken_takes_nothing(credits, wallet):
    turn = await _billed(credits, balance=500)
    wallet.balance = 1

    async def spent_right_after_the_read(user_id: str) -> int:
        # Another run takes the last credit between the read and the debit.
        left, wallet.balance = wallet.balance, 0
        return left

    credits.get_credits.side_effect = spent_right_after_the_read
    await turn.record(25_000)

    assert (turn.charged_credits, wallet.debits, wallet.balance) == (0, [], 0)


@pytest.mark.asyncio
async def test_a_debit_that_keeps_failing_is_never_asked_for_twice(credits, caplog):
    caplog.set_level(logging.WARNING, logger="backend.copilot.credit_billing")
    turn = await _billed(credits, balance=500)
    credits.spend_credits.side_effect = ConnectionError("db down")

    async with credit_billing(turn):
        await record_credit_billed_cost("user-1", 25_000)
        await record_credit_billed_cost("user-1", 20_000)

    # Each charge asks only for what the ones before it didn't, 5 credits in
    # all, and each failure is logged as a possible leak.
    assert _charged(credits) == [2, 2, 1]
    assert turn.charged_credits == 5
    assert [leak.levelno for leak in _leaks(caplog)] == [logging.ERROR] * 3

    # Once debits go through again, a cost landing late charges only its own
    # part.
    credits.spend_credits.side_effect = None
    await turn.record(10_000)
    assert _charged(credits)[3:] == [1]
    assert turn.charged_credits == 6


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

    # 2 as the first cost landed and 1 rounding it up at the end; then 55,000
    # microdollars in all is 6 credits, not 3 + 3.
    assert _charged(credits) == [2, 1, 3]
    assert turn.charged_credits == 6


@pytest.mark.asyncio
async def test_a_balance_spent_meanwhile_pays_only_what_is_left(credits, wallet):
    turn = await _turn()

    async with credit_billing(turn):
        await record_credit_billed_cost("user-1", 400_000)
        assert wallet.debits == [40]
        # Other runs spent all but 20 credits meanwhile.
        wallet.balance = 20
        await record_credit_billed_cost("user-1", 300_000)
        # 30 credits due and 20 left: the turn takes the 20.
        assert wallet.debits == [40, 20]

    # 70 owed in all; the 10 still due when the turn ended found nothing left.
    assert _charged(credits) == [40, 30, 20, 10]
    assert (turn.charged_credits, wallet.balance) == (60, 0)


@pytest.mark.asyncio
async def test_another_users_cost_stays_off_the_turn(credits):
    turn = await _billed(credits, balance=500)

    async with credit_billing(turn):
        assert not await record_credit_billed_cost("user-2", 90_000)

    credits.spend_credits.assert_not_awaited()
