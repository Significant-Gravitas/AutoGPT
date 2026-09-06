"""Tests for backend.copilot.computer: describe without waking, screen on in place."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from e2b import SandboxState

from backend.blocks.desktop._api import DesktopStream
from backend.blocks.desktop._common import SHARED_PATH, WORKSPACE_PATH
from backend.copilot.computer import (
    ComputerInfo,
    computer_owner,
    describe_computer,
    mounts_for,
    open_desktop,
    screen_is_on,
)
from backend.copilot.tools.e2b_sandbox import SandboxOwner

_C = "backend.copilot.computer"
_USER, _EXPERT, _SESSION = "user-1", "exp-1", "sess-1"


def _info(sandbox_id: str, state: SandboxState, mounts: str = "attached"):
    return SimpleNamespace(
        sandbox_id=sandbox_id,
        state=state,
        started_at=datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc),
        cpu_count=1,
        memory_mb=2048,
        template_id="agpt-desktop-1x2",
        metadata={"autogpt_kind": "shell", "autogpt_mounts": mounts},
    )


def _redis(display: str | None):
    r = MagicMock()
    r.get = AsyncMock(return_value=display)
    r.set = AsyncMock()
    r.delete = AsyncMock()
    return r


def _sandbox(sandbox_id: str = "sb-1"):
    sb = MagicMock()
    sb.sandbox_id = sandbox_id
    return sb


def _desktop(sandbox_id: str = "sb-1", mounted: bool = True):
    d = MagicMock()
    d.ensure_display = AsyncMock()
    d.ensure_persistent_home = AsyncMock()
    d.is_workspace_mounted = AsyncMock(return_value=mounted)
    d.start_stream = AsyncMock(
        return_value=DesktopStream(
            url="https://6080-x.e2b.app/vnc.html", sandbox_id=sandbox_id
        )
    )
    return d


class TestComputerOwner:
    def test_expert_session_is_the_experts_computer(self):
        assert computer_owner(_SESSION, _EXPERT) == SandboxOwner(
            kind="expert", id=_EXPERT
        )
        assert computer_owner(_SESSION, None) == SandboxOwner(
            kind="session", id=_SESSION
        )

    def test_mounts_need_a_user(self):
        assert mounts_for(None, _EXPERT) == {}
        assert set(mounts_for(_USER, _EXPERT)) == {WORKSPACE_PATH, SHARED_PATH}
        assert set(mounts_for(_USER, None)) == {WORKSPACE_PATH}


class TestDescribeComputer:
    @pytest.mark.asyncio
    async def test_reports_the_box_and_its_screen_without_connecting(self):
        owner = SandboxOwner(kind="expert", id=_EXPERT)
        with (
            patch(f"{_C}.chat_config") as cfg,
            patch(
                f"{_C}.list_owned_sandboxes",
                AsyncMock(return_value=[_info("sb-1", SandboxState.PAUSED)]),
            ) as list_mock,
            patch(f"{_C}.get_redis_async", AsyncMock(return_value=_redis("sb-1"))),
            patch(f"{_C}.DesktopSession") as desktop_cls,
        ):
            cfg.active_e2b_api_key = "k"
            info = await describe_computer(owner, mounts_for(_USER, _EXPERT))

        assert isinstance(info, ComputerInfo)
        assert info.owner_kind == "expert" and info.e2b_active
        assert info.box and info.box.state == "paused" and info.box.mounts_attached
        assert info.box.cpu_count == 1 and info.box.memory_mb == 2048
        assert info.screen_on is True
        assert info.mounts[SHARED_PATH].startswith("autogpt-user-")
        list_mock.assert_awaited_once_with(owner, "k")
        # Describing must never resume a paused box.
        desktop_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_screen_flag_for_a_replaced_box_does_not_count(self):
        owner = SandboxOwner(kind="session", id=_SESSION)
        with (
            patch(f"{_C}.chat_config") as cfg,
            patch(
                f"{_C}.list_owned_sandboxes",
                AsyncMock(return_value=[_info("sb-new", SandboxState.RUNNING)]),
            ),
            patch(f"{_C}.get_redis_async", AsyncMock(return_value=_redis("sb-old"))),
        ):
            cfg.active_e2b_api_key = "k"
            info = await describe_computer(owner, {})
        assert info.box and info.box.state == "running"
        assert info.screen_on is False

    @pytest.mark.asyncio
    async def test_without_e2b_it_says_so_and_lists_nothing(self):
        owner = SandboxOwner(kind="session", id=_SESSION)
        with (
            patch(f"{_C}.chat_config") as cfg,
            patch(f"{_C}.list_owned_sandboxes", AsyncMock()) as list_mock,
        ):
            cfg.active_e2b_api_key = None
            info = await describe_computer(owner, {})
        assert info.e2b_active is False and info.box is None
        assert info.screen_on is False
        list_mock.assert_not_awaited()


class TestOpenDesktop:
    def _patches(self, redis, sandbox, desktop):
        desktop_cls = MagicMock(return_value=desktop)
        return (
            patch(f"{_C}.get_redis_async", AsyncMock(return_value=redis)),
            patch(f"{_C}.get_or_create_owner_sandbox", AsyncMock(return_value=sandbox)),
            patch(f"{_C}.DesktopSession", desktop_cls),
            patch(f"{_C}.chat_config"),
        )

    @pytest.mark.asyncio
    async def test_turns_the_screen_on_in_the_owners_own_box(self):
        owner = SandboxOwner(kind="expert", id=_EXPERT)
        mounts = mounts_for(_USER, _EXPERT)
        redis, sandbox, desktop = _redis(None), _sandbox("sb-1"), _desktop("sb-1")
        redis_p, get_p, cls_p, cfg_p = self._patches(redis, sandbox, desktop)
        with redis_p, get_p as get_mock, cls_p as desktop_cls, cfg_p as cfg:
            cfg.e2b_sandbox_timeout = 420
            cfg.e2b_sandbox_template = "agpt-desktop-1x2"
            cfg.e2b_sandbox_on_timeout = "pause"
            stream, first_time, shared = await open_desktop(
                owner, mounts, "k", user_id=_USER, session_id=_SESSION
            )

        assert first_time and shared and stream.sandbox_id == "sb-1"
        # The same box a turn would use, found or created the same way, but
        # not counted as a turn so the agent's turn-end pause still fires.
        get_mock.assert_awaited_once_with(
            owner,
            "k",
            timeout=420,
            template="agpt-desktop-1x2",
            on_timeout="pause",
            volume_mounts=mounts,
            user_id=_USER,
            session_id=_SESSION,
            count_turn=False,
        )
        desktop_cls.assert_called_once_with(sandbox)
        desktop.ensure_display.assert_awaited_once()
        desktop.ensure_persistent_home.assert_awaited_once()
        desktop.start_stream.assert_awaited_once()
        assert redis.set.await_args.args[:2] == (
            f"copilot:e2b:expert:{_EXPERT}:shell:display",
            "sb-1",
        )

    @pytest.mark.asyncio
    async def test_second_open_only_refreshes_the_stream(self):
        owner = SandboxOwner(kind="session", id=_SESSION)
        redis, sandbox, desktop = _redis("sb-1"), _sandbox("sb-1"), _desktop("sb-1")
        redis_p, get_p, cls_p, cfg_p = self._patches(redis, sandbox, desktop)
        with redis_p, get_p, cls_p, cfg_p:
            stream, first_time, shared = await open_desktop(owner, {}, "k")

        assert not first_time and stream.sandbox_id == "sb-1"
        # ensure_display is idempotent and cheap; the home redirect is not
        # repeated once the screen has been on.
        desktop.ensure_display.assert_awaited_once()
        desktop.ensure_persistent_home.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_volume_means_nothing_to_redirect(self):
        owner = SandboxOwner(kind="session", id=_SESSION)
        redis, sandbox = _redis(None), _sandbox("sb-1")
        desktop = _desktop("sb-1", mounted=False)
        redis_p, get_p, cls_p, cfg_p = self._patches(redis, sandbox, desktop)
        with redis_p, get_p, cls_p, cfg_p:
            _stream, first_time, shared = await open_desktop(owner, {}, "k")
        assert first_time and not shared
        desktop.ensure_persistent_home.assert_not_awaited()


class TestScreenIsOn:
    @pytest.mark.asyncio
    async def test_matches_only_the_current_box(self):
        owner = SandboxOwner(kind="session", id=_SESSION)
        with patch(f"{_C}.get_redis_async", AsyncMock(return_value=_redis(b"sb-1"))):
            assert await screen_is_on(owner, "sb-1") is True
            assert await screen_is_on(owner, "sb-2") is False
