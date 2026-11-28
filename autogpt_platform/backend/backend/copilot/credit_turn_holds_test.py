"""Credit-billed turns that run at once share their ledger's balance."""

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from backend.copilot import credit_billing
from backend.copilot.config import ChatConfig
from backend.copilot.credit_billing import bill_to_credits
from backend.copilot.credit_billing_test import _CREDITS_ON, OUT_OF_TODAY, _Wallet
from backend.copilot.credit_turn_holds import LedgerHolds
from backend.data import redis_client
from backend.util.testing import is_tcp_port_reachable

pytestmark = pytest.mark.skipif(
    not is_tcp_port_reachable(redis_client.HOST, redis_client.PORT),
    reason="needs Redis",
)


@pytest.fixture
async def holds():
    ledger = LedgerHolds(f"test-{uuid.uuid4()}", lifetime_seconds=60)
    yield ledger
    redis = await redis_client.get_redis_async()
    await redis.delete(ledger.key)


@pytest.mark.asyncio
async def test_a_turn_is_offered_what_the_others_do_not_hold(holds: LedgerHolds):
    assert await _hold(holds, "a", balance=500, credits=300) == 300
    assert await _hold(holds, "b", balance=500, credits=300) == 200
    assert await _hold(holds, "c", balance=500, credits=300) == 0


@pytest.mark.asyncio
async def test_a_top_up_adds_to_the_turns_own_hold(holds: LedgerHolds):
    assert await _hold(holds, "a", balance=500, credits=100) == 100
    assert await _hold(holds, "a", balance=500, credits=150, at_least=1) == 150
    assert await _hold(holds, "b", balance=500, credits=1000) == 250
    assert await _hold(holds, "a", balance=500, credits=150, at_least=1) == 0


class _ReadTogether:
    """Redis whose hash reads answer only once both claims have read."""

    def __init__(self, redis, barrier: asyncio.Barrier) -> None:
        self._redis = redis
        self._barrier = barrier
        self._waited: set[asyncio.Task] = set()

    def __getattr__(self, name: str):
        return getattr(self._redis, name)

    async def hgetall(self, key: str):
        fields = await self._redis.hgetall(key)
        task = asyncio.current_task()
        if task is not None and task not in self._waited:
            self._waited.add(task)
            await asyncio.wait_for(self._barrier.wait(), timeout=10)
        return fields


@pytest.mark.asyncio
async def test_two_turns_that_read_the_holds_at_once_never_share_credits(
    holds: LedgerHolds,
):
    redis = _ReadTogether(await redis_client.get_redis_async(), asyncio.Barrier(2))
    with patch(
        "backend.copilot.credit_turn_holds.get_redis_async",
        AsyncMock(return_value=redis),
    ):
        claims = await asyncio.gather(
            _hold(holds, "a", balance=100, credits=1000),
            _hold(holds, "b", balance=100, credits=1000),
        )
    assert sorted(claims) == [0, 100]


@pytest.mark.asyncio
async def test_a_refused_hold_leaves_nothing_that_never_lapses(holds: LedgerHolds):
    assert await _hold(holds, "a", balance=10, credits=1000) == 0
    redis = await redis_client.get_redis_async()
    assert 0 < await redis.ttl(holds.key) <= holds.lifetime_seconds


@pytest.mark.asyncio
async def test_a_lowered_or_released_hold_frees_its_credits(holds: LedgerHolds):
    assert await _hold(holds, "a", balance=500, credits=1000) == 500
    # The turn has debited 400: the balance dropped by as much as its hold.
    await holds.lower("a", 100)
    assert await _hold(holds, "b", balance=100, credits=1000) == 0
    await holds.release("a")
    assert await _hold(holds, "b", balance=100, credits=1000) == 100


@pytest.mark.asyncio
async def test_a_lapsed_hold_holds_nothing(holds: LedgerHolds):
    redis = await redis_client.get_redis_async()
    await redis.hset(holds.key, "dead-turn", "500:1")
    assert await _hold(holds, "a", balance=500, credits=1000) == 500


async def _hold(
    holds: LedgerHolds, turn_id: str, *, balance: int, credits: int, at_least=50
) -> int:
    return await holds.hold(
        turn_id, read_balance=_reads(balance), credits=credits, at_least=at_least
    )


def _reads(balance: int):
    async def read() -> int:
        return balance

    return read


@pytest.fixture
def ledger_id() -> str:
    return f"test-{uuid.uuid4()}"


@pytest.fixture
async def wallet(mocker, ledger_id: str):
    """100 credits for a user, or for an organization, of which ``ledger_id``
    is the id; its holds are deleted afterwards."""
    wallet = _Wallet(balance=100)
    db = MagicMock()
    db.get_credits = wallet.get_credits
    db.spend_credits = wallet.spend_credits
    db.get_org_credits = AsyncMock(side_effect=lambda org_id: wallet.balance)
    db.spend_org_credits = AsyncMock()
    db.get_personal_org_owner = AsyncMock(return_value=None)
    mocker.patch("backend.copilot.credit_billing.credit_db", return_value=db)
    mocker.patch(_CREDITS_ON, True)
    mocker.patch.object(
        ChatConfig, "thinking_available", new_callable=PropertyMock, return_value=True
    )
    yield wallet
    redis = await redis_client.get_redis_async()
    await redis.delete(LedgerHolds(ledger_id, 1).key)


async def _bill(user_id: str, session_id: str, organization_id: str | None = None):
    return await bill_to_credits(
        user_id=user_id,
        session_id=session_id,
        tool_name="autopilot_block",
        refusal=OUT_OF_TODAY,
        organization_id=organization_id,
    )


@pytest.mark.asyncio
async def test_two_chats_of_one_user_share_its_balance(wallet, ledger_id: str):
    first = await _bill(ledger_id, "chat-a")
    assert first is not None and first.cap_credits == 100
    assert await _bill(ledger_id, "chat-b") is None


@pytest.mark.asyncio
async def test_members_of_one_organization_share_its_balance(wallet, ledger_id: str):
    first = await _bill("member-a", "chat-a", organization_id=ledger_id)
    assert first is not None and first.cap_credits == 100
    assert await _bill("member-b", "chat-b", organization_id=ledger_id) is None


@pytest.mark.asyncio
async def test_a_debit_landing_while_a_turn_starts_is_not_promised_again(
    wallet, ledger_id: str, mocker
):
    first = await _bill(ledger_id, "chat-a")
    assert first is not None and first.cap_credits == 100
    real_balance = credit_billing._balance
    debited = False

    async def first_turn_debits_during_the_read(user_id, organization_id):
        nonlocal debited
        seen = await real_balance(user_id, organization_id)
        if not debited:
            debited = True
            # 99 credits debited, then the first turn's hold lowered to 1.
            await first.record(990_000)
        return seen

    mocker.patch.object(credit_billing, "_balance", first_turn_debits_during_the_read)
    assert await _bill(ledger_id, "chat-b") is None


@pytest.mark.asyncio
async def test_two_turns_topping_up_at_once_never_hold_more_than_the_balance(
    wallet, ledger_id: str
):
    wallet.balance = 300
    first = await _bill(ledger_id, "chat-a")
    second = await _bill(ledger_id, "chat-b")
    assert first is not None and second is not None
    assert (first.held_credits, second.held_credits) == (100, 100)

    # Both read the holds before either writes, each asking for 150 of the
    # 100 left.
    redis = _ReadTogether(await redis_client.get_redis_async(), asyncio.Barrier(2))
    with patch(
        "backend.copilot.credit_turn_holds.get_redis_async",
        AsyncMock(return_value=redis),
    ):
        budgets = await asyncio.gather(first.grow(1.5), second.grow(1.5))

    assert sorted(budgets) == [1.0, 2.0]
    assert sorted([first.held_credits, second.held_credits]) == [100, 200]
    assert first.held_credits + second.held_credits <= wallet.balance


@pytest.mark.asyncio
async def test_a_top_up_landing_after_the_release_leaves_no_hold(
    wallet, ledger_id: str, mocker
):
    wallet.balance = 300
    turn = await _bill(ledger_id, "chat-a")
    assert turn is not None
    ledger = LedgerHolds(ledger_id, 1)
    real_hold = LedgerHolds.hold
    holding, go_on = asyncio.Event(), asyncio.Event()

    async def slow_hold(self, *args, **kwargs):
        holding.set()
        await go_on.wait()
        return await real_hold(self, *args, **kwargs)

    mocker.patch.object(LedgerHolds, "hold", slow_hold)
    # Waits on events; the bound only ends a run that hangs.
    async with asyncio.timeout(30):
        growing = asyncio.create_task(turn.grow(1.5))
        await holding.wait()
        await turn.release()
        go_on.set()
        await growing

    redis = await redis_client.get_redis_async()
    assert set(await redis.hgetall(ledger.key)) == {"version"}
    assert turn.held_credits == 100
