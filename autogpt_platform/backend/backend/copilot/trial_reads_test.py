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
    shared_trial_read,
    sharing_trial_reads,
    trial_reads_shared,
)
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
            "backend.copilot.rate_limit._fetch_user_tier",
            new=AsyncMock(return_value=SubscriptionTier.TRIAL),
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
