"""Tests for the lazy auto-registration helper."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .scheduling import (
    REGISTRATION_KEY_TEMPLATE,
    REGISTRATION_TTL_SECONDS,
    ensure_community_rebuild_scheduled,
)

# Patch at source modules because scheduling.py does lazy imports
# inside the helper (defensive — avoids import-time side effects in
# test envs). The helper's flag check and timezone resolver ARE local
# to the module, so those are patched in-place.
_PATH_REDIS = "backend.data.redis_client.get_redis_async"
_PATH_SCHEDULER = "backend.util.clients.get_scheduler_client"
_PATH_FLAG = "backend.copilot.graphiti.scheduling._is_communities_enabled"
_PATH_TZ = "backend.copilot.graphiti.scheduling._resolve_user_timezone"


def _patches(*, flag=True, setnx_ok=True, tz="UTC"):
    """Common patch stack — returns a list of (target, AsyncMock) tuples.

    `setnx_ok=True`  → SETNX returns truthy (we are the first writer,
                       proceed to register).
    `setnx_ok=False` → SETNX returns falsy (already registered, bail).
    """
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=setnx_ok)
    scheduler = MagicMock()
    scheduler.add_community_rebuild_schedule = AsyncMock(
        return_value={"id": "community_rebuild_abc"}
    )
    return (
        {
            _PATH_FLAG: AsyncMock(return_value=flag),
            _PATH_TZ: AsyncMock(return_value=tz),
            _PATH_REDIS: AsyncMock(return_value=redis),
            _PATH_SCHEDULER: MagicMock(return_value=scheduler),
        },
        redis,
        scheduler,
    )


def _apply(patches: dict):
    """Helper to enter all patches as a single context."""
    from contextlib import ExitStack

    stack = ExitStack()
    for target, m in patches.items():
        if isinstance(m, AsyncMock):
            stack.enter_context(patch(target, new=m))
        else:
            stack.enter_context(patch(target, m))
    return stack


class TestEnsureCommunityRebuildScheduled:
    @pytest.mark.asyncio
    async def test_empty_user_id_is_noop(self) -> None:
        result = await ensure_community_rebuild_scheduled("")
        assert result is None

    @pytest.mark.asyncio
    async def test_flag_off_returns_skipped(self) -> None:
        patches, _, scheduler = _patches(flag=False)
        with _apply(patches):
            result = await ensure_community_rebuild_scheduled("abc")
        assert result == {
            "skipped": True,
            "reason": "graphiti_communities_disabled",
        }
        scheduler.add_community_rebuild_schedule.assert_not_called()

    @pytest.mark.asyncio
    async def test_setnx_already_set_returns_none_and_skips_scheduler(
        self,
    ) -> None:
        patches, _, scheduler = _patches(setnx_ok=False)
        with _apply(patches):
            result = await ensure_community_rebuild_scheduled("abc")
        # SETNX said "already registered" → no work to do
        assert result is None
        scheduler.add_community_rebuild_schedule.assert_not_called()

    @pytest.mark.asyncio
    async def test_setnx_first_writer_registers_with_scheduler(self) -> None:
        patches, redis, scheduler = _patches(tz="America/New_York")
        with _apply(patches):
            result = await ensure_community_rebuild_scheduled("abc")

        # SETNX called with the right key + TTL + nx flag
        redis.set.assert_called_once_with(
            REGISTRATION_KEY_TEMPLATE.format(user_id="abc"),
            "1",
            nx=True,
            ex=REGISTRATION_TTL_SECONDS,
        )
        # Scheduler called with the user_id + resolved timezone
        scheduler.add_community_rebuild_schedule.assert_awaited_once_with(
            user_id="abc", user_timezone="America/New_York"
        )
        # Returns whatever the scheduler returned
        assert result == {"id": "community_rebuild_abc"}

    @pytest.mark.asyncio
    async def test_redis_failure_still_calls_scheduler(self) -> None:
        """If Redis is unavailable, we still register — the scheduler's
        own ``replace_existing=True`` provides the idempotency backstop."""
        patches, _, scheduler = _patches()
        # Override the redis patch to raise on .set
        patches[_PATH_REDIS] = AsyncMock(side_effect=RuntimeError("redis down"))
        with _apply(patches):
            result = await ensure_community_rebuild_scheduled("abc")
        scheduler.add_community_rebuild_schedule.assert_awaited_once()
        assert result == {"id": "community_rebuild_abc"}

    @pytest.mark.asyncio
    async def test_scheduler_failure_returns_skipped_dict_no_raise(self) -> None:
        patches, _, scheduler = _patches()
        scheduler.add_community_rebuild_schedule = AsyncMock(
            side_effect=RuntimeError("scheduler unreachable")
        )
        # Re-patch the scheduler factory to return the broken stub
        patches[_PATH_SCHEDULER] = MagicMock(return_value=scheduler)
        with _apply(patches):
            result = await ensure_community_rebuild_scheduled("abc")
        # MUST NOT raise; surface the failure as a skipped dict
        assert result == {"skipped": True, "reason": "registration_failed"}
