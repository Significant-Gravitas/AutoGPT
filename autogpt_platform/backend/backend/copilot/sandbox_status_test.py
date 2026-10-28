"""The turn that got a new sandbox in place of a gone one is told, once."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.sandbox_status import (
    SANDBOX_REPLACED_NOTICE,
    build_sandbox_replaced_notice,
    sandbox_replaced_key,
)

pytestmark = pytest.mark.asyncio(loop_scope="session")


def _patch_redis(redis: AsyncMock):
    return patch(
        "backend.copilot.sandbox_status.get_redis_async",
        new_callable=AsyncMock,
        return_value=redis,
    )


async def test_a_flagged_turn_gets_the_notice_and_clears_the_flag():
    redis = AsyncMock()
    redis.getdel = AsyncMock(return_value=b"1")
    with _patch_redis(redis):
        notice = await build_sandbox_replaced_notice("sess-1")

    assert notice == SANDBOX_REPLACED_NOTICE
    assert "only `~/workspace`" in notice.lower()
    redis.getdel.assert_awaited_once_with(sandbox_replaced_key("sess-1"))


async def test_an_unflagged_turn_gets_nothing():
    redis = AsyncMock()
    redis.getdel = AsyncMock(return_value=None)
    with _patch_redis(redis):
        assert await build_sandbox_replaced_notice("sess-1") == ""


async def test_a_redis_error_never_fails_the_turn():
    redis = AsyncMock()
    redis.getdel = AsyncMock(side_effect=ConnectionError("down"))
    with _patch_redis(redis):
        assert await build_sandbox_replaced_notice("sess-1") == ""
