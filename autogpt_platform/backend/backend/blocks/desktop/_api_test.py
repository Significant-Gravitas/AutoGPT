"""Desktop client: volumes are retried before they are given up on, every
create has a client-side deadline, and a resumed desktop keeps its timeout."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.desktop._api import (
    WORKSPACE_PATH,
    DesktopSession,
    _create_sandbox_with_volumes,
)

_M = "backend.blocks.desktop._api"
_META = {"service": "autogpt-platform", "autogpt_mounts": "attached"}


@pytest.fixture(autouse=True)
def _no_real_waits():
    with patch(f"{_M}.asyncio.sleep", AsyncMock()):
        yield


@pytest.mark.asyncio
async def test_mounted_create_is_retried_once_before_falling_back():
    box = MagicMock()
    with (
        patch(f"{_M}.resolve_volume", AsyncMock(side_effect=lambda name, key: name)),
        patch(f"{_M}.AsyncSandbox") as cls,
    ):
        cls.create = AsyncMock(side_effect=[RuntimeError("502"), box])
        sandbox, info = await _create_sandbox_with_volumes(
            {WORKSPACE_PATH: "vol-user"}, "k", 900, metadata=_META
        )
    assert sandbox is box and info.volume_mounted and info.warning is None
    assert cls.create.await_count == 2
    assert all("volume_mounts" in c.kwargs for c in cls.create.await_args_list)


@pytest.mark.asyncio
async def test_fallback_after_every_mounted_attempt_says_so_in_the_stamp():
    box = MagicMock()
    with (
        patch(f"{_M}.resolve_volume", AsyncMock(side_effect=lambda name, key: name)),
        patch(f"{_M}.AsyncSandbox") as cls,
    ):
        cls.create = AsyncMock(
            side_effect=[RuntimeError("no volumes"), RuntimeError("no volumes"), box]
        )
        sandbox, info = await _create_sandbox_with_volumes(
            {WORKSPACE_PATH: "vol-user"}, "k", 900, metadata=_META
        )
    assert sandbox is box and not info.volume_mounted and info.warning
    fallback = cls.create.await_args_list[-1].kwargs
    assert "volume_mounts" not in fallback
    # The Computer tab reads this stamp; it must not claim mounts it lacks.
    assert fallback["metadata"]["autogpt_mounts"] == "none"


@pytest.mark.asyncio
async def test_a_create_that_hangs_is_cut_off():
    import asyncio

    async def never(**_kwargs):
        await asyncio.Event().wait()

    with (
        patch(f"{_M}.AsyncSandbox") as cls,
        patch(f"{_M}.CREATE_TIMEOUT_SECONDS", 0.01),
    ):
        cls.create = AsyncMock(side_effect=never)
        with pytest.raises(asyncio.TimeoutError):
            await _create_sandbox_with_volumes(None, "k", 900)


@pytest.mark.asyncio
async def test_connect_rearms_the_running_time_limit():
    with patch(f"{_M}.AsyncSandbox") as cls:
        cls.connect = AsyncMock(return_value=MagicMock())
        await DesktopSession.connect("sb-1", "k", timeout_seconds=900)
    cls.connect.assert_awaited_once_with("sb-1", api_key="k", timeout=900)
