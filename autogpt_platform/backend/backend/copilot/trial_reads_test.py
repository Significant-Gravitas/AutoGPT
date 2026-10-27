"""One trial read per ``trial_reads_shared`` block, and none shared outside it."""

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.chat.routes import StreamChatRequest, stream_chat_post
from backend.copilot.rate_limit import (
    SubscriptionTier,
    check_rate_limit,
    enforce_payment_paywall,
    get_global_rate_limits,
)
from backend.copilot.trial_reads import (
    forget_usage_state,
    shared_trial_read,
    shared_usage_state_read,
    sharing_trial_reads,
    trial_reads_fresh,
    trial_reads_shared,
)
from backend.data.pro_activation import UsageActivationState
from backend.data.subscription_trial import TrialState


def _trial(*, active: bool = True, cost: int = 0) -> MagicMock:
    trial = MagicMock(spec=TrialState)
    trial.active = active
    trial.cost_microdollars = cost
    trial.offer = MagicMock(
        daily_cost_limit=1_000_000,
        weekly_cost_limit=5_000_000,
        total_cost_limit=10_000_000,
    )
    trial.ends_at = None
    return trial


@pytest.mark.asyncio
async def test_outside_a_block_every_call_reads():
    read = AsyncMock(return_value=_trial())
    await shared_trial_read("u1", read)
    await shared_trial_read("u1", read)
    assert read.await_count == 2


@pytest.mark.asyncio
async def test_inside_a_block_a_user_is_read_once():
    trial = _trial()
    read = AsyncMock(return_value=trial)
    other = AsyncMock(return_value=None)
    with trial_reads_shared():
        assert await shared_trial_read("u1", read) is trial
        assert await shared_trial_read("u1", read) is trial
        assert await shared_trial_read("u2", other) is None
    assert read.await_count == 1
    assert other.await_count == 1
    await shared_trial_read("u1", read)
    assert read.await_count == 2


@pytest.mark.asyncio
async def test_concurrent_checks_wait_on_the_same_read():
    started = 0
    release = asyncio.Event()

    async def read() -> MagicMock:
        nonlocal started
        started += 1
        await release.wait()
        return _trial()

    with trial_reads_shared():
        waiters = [asyncio.create_task(shared_trial_read("u1", read)) for _ in range(3)]
        await asyncio.sleep(0)
        release.set()
        results = await asyncio.gather(*waiters)
    assert started == 1
    assert results[0] is results[1] is results[2]


@pytest.mark.asyncio
async def test_one_caller_giving_up_does_not_cancel_the_shared_read():
    release = asyncio.Event()
    trial = _trial()

    async def read() -> MagicMock:
        await release.wait()
        return trial

    with trial_reads_shared():
        first = asyncio.create_task(shared_trial_read("u1", read))
        second = asyncio.create_task(shared_trial_read("u1", read))
        await asyncio.sleep(0)
        first.cancel()
        release.set()
        assert await second is trial
        with pytest.raises(asyncio.CancelledError):
            await first


@pytest.mark.asyncio
async def test_a_task_started_in_a_block_reads_afresh_once_it_ends():
    read = AsyncMock(return_value=_trial())
    block_ended = asyncio.Event()

    async def background() -> None:
        await shared_trial_read("u1", read)
        await block_ended.wait()
        await shared_trial_read("u1", read)

    with trial_reads_shared():
        task = asyncio.create_task(background())
        await shared_trial_read("u1", read)
    block_ended.set()
    await task
    assert read.await_count == 2


@pytest.mark.asyncio
async def test_a_closed_block_reads_afresh():
    read = AsyncMock(return_value=_trial())
    with trial_reads_shared() as shared:
        await shared_trial_read("u1", read)
        shared.close()
        await shared_trial_read("u1", read)
    assert read.await_count == 2


@pytest.mark.asyncio
async def test_a_fresh_block_reads_every_time_inside_a_shared_one():
    read = AsyncMock(return_value=_trial())
    with trial_reads_shared():
        await shared_trial_read("u1", read)
        with trial_reads_fresh():
            await shared_trial_read("u1", read)
            await shared_trial_read("u1", read)
        await shared_trial_read("u1", read)
    assert read.await_count == 3


@pytest.mark.asyncio
async def test_a_wrapped_handler_shares_one_read_per_call():
    read = AsyncMock(return_value=_trial())

    async def handler(user_id: str, *, limit: int = 1) -> None:
        await shared_trial_read(user_id, read)
        await shared_trial_read(user_id, read)

    wrapped = sharing_trial_reads(handler)
    await wrapped("u1")
    await wrapped("u1")

    assert read.await_count == 2
    assert inspect.signature(wrapped) == inspect.signature(handler)


class _StopTheTurn(Exception):
    pass


@pytest.mark.asyncio
async def test_the_stream_route_runs_its_checks_in_one_block():
    read = AsyncMock(return_value=_trial())

    async def first_checks(session_id: str, user_id: str) -> None:
        await shared_trial_read(user_id, read)
        await shared_trial_read(user_id, read)
        raise _StopTheTurn

    with (
        patch(
            "backend.api.features.chat.routes._validate_and_get_writable_session",
            new=first_checks,
        ),
        pytest.raises(_StopTheTurn),
    ):
        await stream_chat_post(
            "session-1",
            StreamChatRequest(message="hi"),
            "u1",
            MagicMock(),
            http_request=MagicMock(),
        )

    assert read.await_count == 1


@pytest.mark.asyncio
async def test_a_failed_read_fails_every_check_that_shares_it():
    read = AsyncMock(side_effect=ConnectionError("db down"))
    with trial_reads_shared():
        for _ in range(2):
            with pytest.raises(ConnectionError):
                await shared_trial_read("u1", read)
    assert read.await_count == 1


@pytest.mark.asyncio
async def test_the_chat_routes_trial_checks_read_the_trial_once():
    """Paywall, limits and limit check each read a trial user's trial: four
    database round trips before, one inside the route's block."""

    async def run_checks() -> None:
        await enforce_payment_paywall("trial-user")
        daily, weekly, _ = await get_global_rate_limits("trial-user", 1, 1)
        await check_rate_limit("trial-user", daily, weekly)

    store = MagicMock()
    store.get_subscription_trial = AsyncMock(return_value=_trial())
    redis = MagicMock()
    redis.get = AsyncMock(return_value=None)
    with (
        patch(
            "backend.copilot.rate_limit.get_ready_usage_state",
            new=AsyncMock(
                return_value=UsageActivationState(
                    user_id="trial-user",
                    tier=SubscriptionTier.TRIAL,
                    trial_id="trial-1",
                    ready=True,
                )
            ),
        ),
        patch(
            "backend.copilot.rate_limit._fetch_cost_limits_flag",
            new=AsyncMock(return_value=None),
        ),
        patch("backend.copilot.rate_limit.credit_db", return_value=store),
        patch(
            "backend.copilot.rate_limit.get_redis_async",
            new=AsyncMock(return_value=redis),
        ),
    ):
        await run_checks()
        unshared = store.get_subscription_trial.await_count
        store.get_subscription_trial.reset_mock()
        with trial_reads_shared():
            await run_checks()
        shared = store.get_subscription_trial.await_count

    assert unshared == 4
    assert shared == 1


def _usage_state(tier: SubscriptionTier = SubscriptionTier.PRO) -> UsageActivationState:
    return UsageActivationState(user_id="u1", tier=tier, ready=True)


@pytest.mark.asyncio
async def test_inside_a_block_a_users_usage_state_is_read_once():
    state = _usage_state()
    read = AsyncMock(return_value=state)
    await shared_usage_state_read("u1", read)
    with trial_reads_shared():
        assert await shared_usage_state_read("u1", read) is state
        assert await shared_usage_state_read("u1", read) is state
        with trial_reads_fresh():
            await shared_usage_state_read("u1", read)
    assert read.await_count == 3


@pytest.mark.asyncio
async def test_a_forgotten_usage_state_is_read_again_and_shared_from_then_on():
    before, after = _usage_state(SubscriptionTier.NO_TIER), _usage_state()
    read = AsyncMock(side_effect=[before, after])
    with trial_reads_shared():
        assert await shared_usage_state_read("u1", read) is before
        forget_usage_state("u1")
        assert await shared_usage_state_read("u1", read) is after
        assert await shared_usage_state_read("u1", read) is after
    assert read.await_count == 2


@pytest.mark.asyncio
async def test_a_forgotten_block_reads_both_again_and_shares_them_from_then_on():
    trial_before, trial_after = _trial(), _trial(active=False)
    state_before = _usage_state(SubscriptionTier.TRIAL)
    state_after = _usage_state()
    read_trial = AsyncMock(side_effect=[trial_before, trial_after])
    read_state = AsyncMock(side_effect=[state_before, state_after])
    with trial_reads_shared() as shared:
        assert await shared_trial_read("u1", read_trial) is trial_before
        assert await shared_usage_state_read("u1", read_state) is state_before
        shared.forget()
        for _ in range(2):
            assert await shared_trial_read("u1", read_trial) is trial_after
            assert await shared_usage_state_read("u1", read_state) is state_after
    assert read_trial.await_count == 2
    assert read_state.await_count == 2


@pytest.mark.asyncio
async def test_a_failed_usage_state_read_fails_every_check_in_its_block_only():
    read = AsyncMock(side_effect=[ConnectionError("db down"), _usage_state()])
    with trial_reads_shared():
        for _ in range(2):
            with pytest.raises(ConnectionError):
                await shared_usage_state_read("u1", read)
    state = await shared_usage_state_read("u1", read)
    assert state.tier == SubscriptionTier.PRO
    assert read.await_count == 2


@pytest.mark.asyncio
async def test_the_chat_routes_checks_read_the_usage_state_once():
    """Paywall, limits and limit check each read a paid user's usage state:
    three database round trips outside a block, one inside the route's."""

    async def run_checks() -> None:
        await enforce_payment_paywall("u1")
        daily, weekly, tier = await get_global_rate_limits("u1", 1, 1)
        await check_rate_limit("u1", daily, weekly, expected_tier=tier)

    usage_db = MagicMock()
    usage_db.get_usage_activation_state = AsyncMock(return_value=_usage_state())
    redis = MagicMock()
    redis.get = AsyncMock(return_value=None)
    with (
        patch(
            "backend.copilot.usage_activation.pro_activation_db", return_value=usage_db
        ),
        patch(
            "backend.copilot.rate_limit._fetch_cost_limits_flag",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "backend.copilot.rate_limit.get_feature_flag_value",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "backend.copilot.rate_limit.is_feature_enabled",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "backend.copilot.rate_limit.get_redis_async",
            new=AsyncMock(return_value=redis),
        ),
    ):
        await run_checks()
        unshared = usage_db.get_usage_activation_state.await_count
        usage_db.get_usage_activation_state.reset_mock()
        with trial_reads_shared():
            await run_checks()
        shared = usage_db.get_usage_activation_state.await_count

    assert unshared == 3
    assert shared == 1
