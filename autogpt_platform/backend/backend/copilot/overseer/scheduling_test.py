"""Registration is one Redis ``SET NX`` — two racing turns yield one cron."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.overseer.scheduling import (
    OVERSEER_REGISTRATION_PREFIX,
    ensure_task_overseer_scheduled,
)

_MODULE = "backend.copilot.overseer.scheduling"


def _redis(*, claimed: bool) -> MagicMock:
    redis = MagicMock()
    redis.set = AsyncMock(return_value=True if claimed else None)
    redis.delete = AsyncMock()
    return redis


def _patches(redis: MagicMock, scheduler: MagicMock):
    return (
        patch(f"{_MODULE}.is_feature_enabled", AsyncMock(return_value=True)),
        patch(f"{_MODULE}.get_redis_async", AsyncMock(return_value=redis)),
        patch("backend.util.clients.get_scheduler_client", return_value=scheduler),
    )


@pytest.mark.asyncio
async def test_first_claim_registers_the_cron():
    redis = _redis(claimed=True)
    scheduler = MagicMock(add_task_overseer_schedule=AsyncMock())
    flag, r, c = _patches(redis, scheduler)
    with flag, r, c:
        await ensure_task_overseer_scheduled("user-1")

    scheduler.add_task_overseer_schedule.assert_awaited_once_with(user_id="user-1")
    assert redis.set.await_args.args[0] == f"{OVERSEER_REGISTRATION_PREFIX}:user-1"
    assert redis.set.await_args.kwargs["nx"] is True


@pytest.mark.asyncio
async def test_losing_the_claim_never_touches_the_scheduler():
    """The loser must not call the scheduler: re-adding the job would reset
    the winner's pending ``next_run_time``."""
    scheduler = MagicMock(add_task_overseer_schedule=AsyncMock())
    flag, r, c = _patches(_redis(claimed=False), scheduler)
    with flag, r, c:
        await ensure_task_overseer_scheduled("user-1")

    scheduler.add_task_overseer_schedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_registration_releases_the_claim():
    redis = _redis(claimed=True)
    scheduler = MagicMock(
        add_task_overseer_schedule=AsyncMock(side_effect=RuntimeError("rpc down"))
    )
    flag, r, c = _patches(redis, scheduler)
    with flag, r, c:
        await ensure_task_overseer_scheduled("user-1")

    redis.delete.assert_awaited_once_with(f"{OVERSEER_REGISTRATION_PREFIX}:user-1")
