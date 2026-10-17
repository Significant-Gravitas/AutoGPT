"""Desktop client: volumes are retried before they are given up on, every
create has a client-side deadline, and a resumed desktop keeps its timeout."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from e2b.exceptions import SandboxException

from backend.blocks.desktop import _api
from backend.blocks.desktop._api import (
    WORKSPACE_PATH,
    DesktopSession,
    PersistenceInfo,
    _create_sandbox_with_volumes,
    is_volume_mounts_disabled,
    mark_volume_mounts_disabled,
)
from backend.util.e2b_network import EgressOwner

_M = "backend.blocks.desktop._api"
_META = {"service": "autogpt-platform", "autogpt_mounts": "attached"}
_OWNER = EgressOwner(kind="session", id="sess-1", user_id="user-1")
_NOT_ENABLED = SandboxException("400: Volume mounts are not enabled.", status_code=400)


@pytest.fixture(autouse=True)
def _no_real_waits():
    with patch(f"{_M}.asyncio.sleep", AsyncMock()):
        yield


@pytest.fixture(autouse=True)
def _no_remembered_refusal():
    _api._volume_mounts_disabled_until.clear()
    yield
    _api._volume_mounts_disabled_until.clear()


def test_only_the_not_enabled_refusal_means_the_team_has_no_mounts():
    assert is_volume_mounts_disabled(_NOT_ENABLED)
    assert not is_volume_mounts_disabled(
        SandboxException("400: template not found", status_code=400)
    )
    assert not is_volume_mounts_disabled(
        SandboxException("500: Volume mounts are not enabled.", status_code=500)
    )
    assert not is_volume_mounts_disabled(RuntimeError("Volume mounts are not enabled"))


@pytest.mark.asyncio
async def test_a_team_without_volume_mounts_falls_back_after_one_refusal():
    box = MagicMock()
    sleep = AsyncMock()
    with (
        patch(f"{_M}.resolve_volume", AsyncMock(side_effect=lambda name, key: name)),
        patch(f"{_M}.asyncio.sleep", sleep),
        patch(f"{_M}.AsyncSandbox") as cls,
    ):
        cls.create = AsyncMock(side_effect=[_NOT_ENABLED, box])
        sandbox, info = await _create_sandbox_with_volumes(
            {WORKSPACE_PATH: "vol-user"}, "k", 900, metadata=_META, owner=_OWNER
        )
    assert sandbox is box and not info.volume_mounted and info.warning
    assert cls.create.await_count == 2
    assert "volume_mounts" not in cls.create.await_args_list[-1].kwargs
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_remembered_refusal_skips_the_mounts_entirely():
    mark_volume_mounts_disabled("k")
    box = MagicMock()
    resolve = AsyncMock(side_effect=lambda name, key: name)
    with (
        patch(f"{_M}.resolve_volume", resolve),
        patch(f"{_M}.AsyncSandbox") as cls,
    ):
        cls.create = AsyncMock(return_value=box)
        sandbox, info = await _create_sandbox_with_volumes(
            {WORKSPACE_PATH: "vol-user"}, "k", 900, metadata=_META, owner=_OWNER
        )
    assert sandbox is box and not info.volume_mounted
    assert info.warning and "not enabled" in info.warning
    resolve.assert_not_awaited()
    assert cls.create.await_count == 1
    only = cls.create.await_args.kwargs
    assert "volume_mounts" not in only
    assert only["metadata"]["autogpt_mounts"] == "none"


@pytest.mark.asyncio
async def test_mounted_create_is_retried_once_before_falling_back():
    box = MagicMock()
    with (
        patch(f"{_M}.resolve_volume", AsyncMock(side_effect=lambda name, key: name)),
        patch(f"{_M}.AsyncSandbox") as cls,
    ):
        cls.create = AsyncMock(side_effect=[RuntimeError("502"), box])
        sandbox, info = await _create_sandbox_with_volumes(
            {WORKSPACE_PATH: "vol-user"}, "k", 900, metadata=_META, owner=_OWNER
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
            {WORKSPACE_PATH: "vol-user"}, "k", 900, metadata=_META, owner=_OWNER
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
            await _create_sandbox_with_volumes(None, "k", 900, owner=_OWNER)


@pytest.mark.asyncio
async def test_connect_rearms_the_running_time_limit():
    box = MagicMock()
    with (
        patch(f"{_M}.AsyncSandbox") as cls,
        patch(f"{_M}.take_baseline", AsyncMock()) as baseline,
    ):
        cls.connect = AsyncMock(return_value=box)
        await DesktopSession.connect("sb-1", "k", timeout_seconds=900, owner=_OWNER)
    cls.connect.assert_awaited_once_with("sb-1", api_key="k", timeout=900)
    # One made before login baselines existed gets its baseline here.
    baseline.assert_awaited_once_with(box, only_if_missing=True)


@pytest.mark.asyncio
async def test_a_create_cancelled_during_its_baseline_kills_the_box():
    """The box is on the meter and no caller has its id yet."""
    import asyncio

    box = MagicMock()
    with (
        patch(
            f"{_M}._create_sandbox_with_volumes",
            AsyncMock(return_value=(box, PersistenceInfo())),
        ),
        patch(f"{_M}.take_baseline", AsyncMock(side_effect=asyncio.CancelledError)),
        patch(f"{_M}.kill_sandbox", AsyncMock()) as kill,
    ):
        with pytest.raises(asyncio.CancelledError):
            await DesktopSession.create("k", 900, 1280, 720, owner=_OWNER)
    kill.assert_awaited_once_with(box)


@pytest.mark.asyncio
async def test_a_desktop_takes_its_login_baseline_before_its_first_command():
    """Without one every internal command refuses, the display's included."""
    order: list[str] = []
    box = MagicMock()
    with (
        patch(
            f"{_M}._create_sandbox_with_volumes",
            AsyncMock(return_value=(box, PersistenceInfo())),
        ),
        patch(
            f"{_M}.take_baseline",
            AsyncMock(side_effect=lambda *_a, **_k: order.append("baseline")),
        ),
        patch.object(
            DesktopSession,
            "ensure_display",
            AsyncMock(side_effect=lambda *_a: order.append("display")),
        ),
        patch.object(DesktopSession, "run_command", AsyncMock()),
    ):
        await DesktopSession.create("k", 900, 1280, 720, owner=_OWNER)
    assert order[:2] == ["baseline", "display"]
