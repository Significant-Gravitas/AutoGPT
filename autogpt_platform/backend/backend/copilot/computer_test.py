"""Tests for backend.copilot.computer: describe without waking, open by owner."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from e2b import SandboxState

from backend.blocks.desktop._api import DesktopStream, PersistenceInfo
from backend.blocks.desktop._common import SHARED_PATH, WORKSPACE_PATH
from backend.copilot.computer import (
    ComputerInfo,
    computer_owner,
    describe_computer,
    mounts_for,
    open_desktop,
)
from backend.copilot.tools.e2b_sandbox import SandboxOwner
from backend.util.sandbox_metadata import deployment_env

_C = "backend.copilot.computer"
_USER, _EXPERT, _SESSION = "user-1", "exp-1", "sess-1"


def _info(kind: str, sandbox_id: str, state: SandboxState, mounts: str = "attached"):
    return SimpleNamespace(
        sandbox_id=sandbox_id,
        state=state,
        started_at=datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc),
        cpu_count=2,
        memory_mb=4096,
        template_id="desktop" if kind == "desktop" else "base",
        metadata={"autogpt_kind": kind, "autogpt_mounts": mounts},
    )


class TestComputerOwner:
    def test_expert_session_is_the_experts_computer(self):
        assert computer_owner(_SESSION, _EXPERT) == SandboxOwner(
            kind="expert", id=_EXPERT
        )
        assert computer_owner(_SESSION, None) == SandboxOwner(
            kind="session", id=_SESSION
        )

    def test_mounts_follow_the_shells_rule(self):
        # An expert always has its home; the shared volume needs a user.
        assert set(mounts_for(None, _EXPERT)) == {WORKSPACE_PATH}
        assert set(mounts_for(_USER, _EXPERT)) == {WORKSPACE_PATH, SHARED_PATH}
        assert set(mounts_for(_USER, None)) == {WORKSPACE_PATH}
        assert mounts_for(None, None) == {}


class TestDescribeComputer:
    @pytest.mark.asyncio
    async def test_reports_both_boxes_without_connecting(self):
        owner = SandboxOwner(kind="expert", id=_EXPERT)
        listed = {
            "shell": [_info("shell", "sb-shell", SandboxState.PAUSED)],
            "desktop": [
                _info("desktop", "sb-desk", SandboxState.RUNNING, mounts="none")
            ],
        }
        list_mock = AsyncMock(side_effect=lambda o, kind, key: listed[kind])
        with (
            patch(f"{_C}.chat_config") as cfg,
            patch(f"{_C}.list_owned_sandboxes", list_mock),
            patch(f"{_C}.DesktopSession") as desktop_cls,
        ):
            cfg.active_e2b_api_key = "k"
            info = await describe_computer(owner, mounts_for(_USER, _EXPERT))

        assert isinstance(info, ComputerInfo)
        assert info.owner_kind == "expert" and info.e2b_active
        assert (
            info.shell and info.shell.state == "paused" and info.shell.mounts_attached
        )
        assert info.desktop and info.desktop.state == "running"
        assert info.desktop.mounts_attached is False
        assert info.mounts[SHARED_PATH].startswith("autogpt-user-")
        # Describing must never resume a paused box.
        desktop_cls.connect.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_failed_listing_shows_nothing_rather_than_failing(self):
        from backend.copilot.tools.e2b_sandbox import SandboxLookupError

        with (
            patch(f"{_C}.chat_config") as cfg,
            patch(
                f"{_C}.list_owned_sandboxes",
                AsyncMock(side_effect=SandboxLookupError("e2b down")),
            ),
        ):
            cfg.active_e2b_api_key = "k"
            info = await describe_computer(SandboxOwner(kind="expert", id=_EXPERT), {})
        assert info.e2b_active and info.shell is None and info.desktop is None

    @pytest.mark.asyncio
    async def test_without_e2b_it_says_so_and_lists_nothing(self):
        owner = SandboxOwner(kind="session", id=_SESSION)
        with (
            patch(f"{_C}.chat_config") as cfg,
            patch(f"{_C}.list_owned_sandboxes", AsyncMock()) as list_mock,
        ):
            cfg.active_e2b_api_key = None
            info = await describe_computer(owner, {})
        assert info.e2b_active is False and info.shell is None and info.desktop is None
        list_mock.assert_not_awaited()


def _redis(stored: str | None, lock_free: bool = True):
    r = MagicMock()
    r.get = AsyncMock(return_value=stored)
    r.set = AsyncMock(return_value=lock_free)
    r.delete = AsyncMock()
    r.eval = AsyncMock(return_value=1)
    return r


def _desktop(sandbox_id="sb-desk"):
    d = MagicMock()
    d.sandbox_id = sandbox_id
    d.ensure_display = AsyncMock()
    d.start_stream = AsyncMock(
        return_value=DesktopStream(
            url="https://6080-x.e2b.app/vnc.html", sandbox_id=sandbox_id
        )
    )
    d.is_workspace_mounted = AsyncMock(return_value=True)
    return d


class TestOpenDesktop:
    @pytest.mark.asyncio
    async def test_a_second_opener_waits_and_reattaches(self):
        """Start from the panel and start_desktop from the model can both miss
        the cache; only one may create, the other reattaches to its box."""
        owner = SandboxOwner(kind="expert", id=_EXPERT)
        redis = _redis(None)
        # Lock taken on the first attempt; free on the second, by which time
        # the first opener has cached its box.
        redis.set = AsyncMock(side_effect=[False, True, True])
        redis.get = AsyncMock(side_effect=[b"sb-first"])
        desktop = _desktop("sb-first")
        with (
            patch(f"{_C}.get_redis_async", AsyncMock(return_value=redis)),
            patch(f"{_C}.asyncio.sleep", AsyncMock()) as sleep,
            patch(f"{_C}.DesktopSession") as desktop_cls,
            patch(f"{_C}.chat_config") as cfg,
        ):
            cfg.e2b_desktop_timeout = 900
            desktop_cls.connect = AsyncMock(return_value=desktop)
            desktop_cls.create = AsyncMock()
            stream, created, _ = await open_desktop(owner, {}, "k")

        assert not created and stream.sandbox_id == "sb-first"
        sleep.assert_awaited_once()
        desktop_cls.create.assert_not_awaited()
        # The lock is released by token, never a bare delete of the key.
        script, _, lock_key, token = redis.eval.await_args.args
        assert lock_key == f"copilot:e2b:expert:{_EXPERT}:desktop:lock"
        assert token == redis.set.await_args_list[1].args[1]

    @pytest.mark.asyncio
    async def test_the_open_is_cut_off_before_the_lock_can_lapse(self):
        owner = SandboxOwner(kind="expert", id=_EXPERT)
        redis = _redis(None)

        async def never(**_kwargs):
            await asyncio.Event().wait()

        with (
            patch(f"{_C}.get_redis_async", AsyncMock(return_value=redis)),
            patch(f"{_C}.find_owned_sandbox_id", AsyncMock(return_value=None)),
            patch(f"{_C}.DesktopSession") as desktop_cls,
            patch(f"{_C}.chat_config") as cfg,
            patch(f"{_C}._DESKTOP_OPEN_DEADLINE_SECONDS", 0.01),
        ):
            cfg.e2b_desktop_timeout = 900
            cfg.e2b_desktop_template = "desktop"
            desktop_cls.create = AsyncMock(side_effect=never)
            with pytest.raises(asyncio.TimeoutError):
                await open_desktop(owner, {}, "k")
        from backend.copilot import computer

        assert (
            computer._DESKTOP_OPEN_DEADLINE_SECONDS < computer._DESKTOP_LOCK_TTL_SECONDS
        )
        # The lock is still released on the way out.
        redis.eval.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_failure_after_reconnect_is_an_error_not_a_new_box(self):
        """Only a failed connect means the box is gone.  A display or stream
        failure on a live box must surface, not abandon the box and bill for
        a second one."""
        owner = SandboxOwner(kind="expert", id=_EXPERT)
        redis = _redis("sb-live")
        desktop = _desktop("sb-live")
        desktop.ensure_display = AsyncMock(side_effect=RuntimeError("no display"))
        with (
            patch(f"{_C}.get_redis_async", AsyncMock(return_value=redis)),
            patch(f"{_C}.DesktopSession") as desktop_cls,
            patch(f"{_C}.chat_config") as cfg,
        ):
            cfg.e2b_desktop_timeout = 900
            desktop_cls.connect = AsyncMock(return_value=desktop)
            desktop_cls.create = AsyncMock()
            with pytest.raises(RuntimeError, match="no display"):
                await open_desktop(owner, {}, "k")
        desktop_cls.create.assert_not_awaited()
        redis.delete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_session_desktop_whose_id_cannot_be_saved_is_killed(self):
        """Nothing can find a session desktop later; an unsaved one would bill
        until timeout.  An expert's is recoverable by metadata and is kept."""
        session_owner = SandboxOwner(kind="session", id=_SESSION)
        redis = _redis(None)
        redis.set = AsyncMock(side_effect=[True, ConnectionError("redis down")])
        desktop = _desktop("sb-new")
        desktop.kill = AsyncMock()
        with (
            patch(f"{_C}.get_redis_async", AsyncMock(return_value=redis)),
            patch(f"{_C}.DesktopSession") as desktop_cls,
            patch(f"{_C}.chat_config") as cfg,
        ):
            cfg.e2b_desktop_timeout = 900
            cfg.e2b_desktop_template = "desktop"
            desktop_cls.create = AsyncMock(return_value=(desktop, PersistenceInfo()))
            with pytest.raises(ConnectionError):
                await open_desktop(session_owner, {}, "k")
        desktop.kill.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_gives_up_when_the_lock_never_frees(self):
        owner = SandboxOwner(kind="expert", id=_EXPERT)
        redis = _redis(None, lock_free=False)
        with (
            patch(f"{_C}.get_redis_async", AsyncMock(return_value=redis)),
            patch(f"{_C}.asyncio.sleep", AsyncMock()),
            patch(f"{_C}.DesktopSession") as desktop_cls,
        ):
            desktop_cls.create = AsyncMock()
            with pytest.raises(RuntimeError, match="still opening"):
                await open_desktop(owner, {}, "k")
        desktop_cls.create.assert_not_awaited()
        redis.eval.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_creates_under_the_owner_key_with_the_owner_mounts(self):
        owner = SandboxOwner(kind="expert", id=_EXPERT)
        mounts = mounts_for(_USER, _EXPERT)
        redis = _redis(None)
        with (
            patch(f"{_C}.get_redis_async", AsyncMock(return_value=redis)),
            patch(f"{_C}.find_owned_sandbox_id", AsyncMock(return_value=None)),
            patch(f"{_C}.DesktopSession") as desktop_cls,
            patch(f"{_C}.chat_config") as cfg,
        ):
            cfg.e2b_desktop_timeout = 900
            cfg.e2b_desktop_template = "desktop"
            desktop_cls.create = AsyncMock(
                return_value=(_desktop(), PersistenceInfo(volume_mounted=True))
            )
            stream, created, shared = await open_desktop(owner, mounts, "k")

        assert created and shared and stream.sandbox_id == "sb-desk"
        kwargs = desktop_cls.create.await_args.kwargs
        assert kwargs["volume_mounts"] == mounts
        assert kwargs["metadata"] == {
            "service": "autogpt-platform",
            "autogpt_owner": f"expert:{_EXPERT}",
            "autogpt_kind": "desktop",
            "autogpt_source": "copilot",
            "autogpt_env": deployment_env(),
            "autogpt_expert": _EXPERT,
            "autogpt_template": "desktop",
            "autogpt_mounts": "attached",
        }
        assert redis.set.await_args.args[0] == f"copilot:e2b:expert:{_EXPERT}:desktop"

    @pytest.mark.asyncio
    async def test_resumes_the_recovered_box_and_reuses_its_stream(self):
        owner = SandboxOwner(kind="expert", id=_EXPERT)
        redis = _redis(None)
        desktop = _desktop("sb-old")
        with (
            patch(f"{_C}.get_redis_async", AsyncMock(return_value=redis)),
            patch(f"{_C}.find_owned_sandbox_id", AsyncMock(return_value="sb-old")),
            patch(f"{_C}.DesktopSession") as desktop_cls,
            patch(f"{_C}.chat_config") as cfg,
        ):
            cfg.e2b_desktop_timeout = 900
            desktop_cls.connect = AsyncMock(return_value=desktop)
            desktop_cls.create = AsyncMock()
            stream, created, shared = await open_desktop(owner, {}, "k")

        assert not created and shared and stream.sandbox_id == "sb-old"
        # A resumed desktop gets the same running-time limit as a new one.
        desktop_cls.connect.assert_awaited_once_with("sb-old", "k", timeout_seconds=900)
        desktop_cls.create.assert_not_awaited()
        desktop.start_stream.assert_awaited_once()
