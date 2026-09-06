"""Tests for StartDesktopTool — the screen goes on in the owner's own box."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.desktop._api import SHARED_PATH, WORKSPACE_PATH, DesktopStream
from backend.blocks.desktop._common import expert_volume_name, user_volume_name

from ._test_data import make_session
from .models import DesktopStreamToolResponse, ErrorResponse
from .start_desktop import StartDesktopTool

_USER = "user-start-desktop-test"
_BOX = "sbx-1"
_STREAM = DesktopStream(
    url="https://6080-sbx.e2b.app/vnc.html?autoconnect=true", sandbox_id=_BOX
)
_C = "backend.copilot.computer"


def _make_redis(display: str | None = None) -> MagicMock:
    redis = MagicMock()
    redis.get = AsyncMock(return_value=display)
    redis.set = AsyncMock()
    redis.delete = AsyncMock()
    return redis


def _make_desktop(mounted: bool = True) -> MagicMock:
    desktop = MagicMock()
    desktop.ensure_display = AsyncMock()
    desktop.ensure_persistent_home = AsyncMock()
    desktop.is_workspace_mounted = AsyncMock(return_value=mounted)
    desktop.start_stream = AsyncMock(return_value=_STREAM)
    return desktop


def _sandbox() -> MagicMock:
    sb = MagicMock()
    sb.sandbox_id = _BOX
    return sb


class _Box:
    """Patches that stand in for the owner's box and the display on it."""

    def __init__(self, display: str | None = None, mounted: bool = True):
        self.redis = _make_redis(display)
        self.sandbox = _sandbox()
        self.desktop = _make_desktop(mounted)
        self.get_box = AsyncMock(return_value=self.sandbox)

    def patches(self):
        return (
            patch(f"{_C}.get_redis_async", AsyncMock(return_value=self.redis)),
            patch(f"{_C}.get_or_create_owner_sandbox", self.get_box),
            patch(f"{_C}.DesktopSession", MagicMock(return_value=self.desktop)),
            patch(f"{_C}.chat_config"),
            patch("backend.copilot.tools.start_desktop.chat_config"),
        )


async def _run(tool, box: _Box, *, user_id, session):
    redis_p, get_p, cls_p, computer_cfg, tool_cfg = box.patches()
    with redis_p, get_p, cls_p, computer_cfg as ccfg, tool_cfg as tcfg:
        tcfg.active_e2b_api_key = "e2b_test_key"
        ccfg.e2b_sandbox_timeout = 420
        ccfg.e2b_sandbox_template = "agpt-desktop-1x2"
        ccfg.e2b_sandbox_on_timeout = "pause"
        return await tool._execute(user_id=user_id, session=session)


class TestStartDesktop:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_unconfigured_e2b_returns_error(self):
        tool = StartDesktopTool()
        session = make_session(user_id=_USER)
        with patch("backend.copilot.tools.start_desktop.chat_config") as mock_config:
            mock_config.active_e2b_api_key = None
            result = await tool._execute(user_id=_USER, session=session)
        assert isinstance(result, ErrorResponse)
        assert result.error == "e2b_unconfigured"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_turns_the_screen_on_in_the_sessions_box(self):
        tool = StartDesktopTool()
        session = make_session(user_id=_USER)
        box = _Box()

        result = await _run(tool, box, user_id=_USER, session=session)

        assert isinstance(result, DesktopStreamToolResponse)
        assert result.desktop_stream["kind"] == "desktop_stream"
        assert result.desktop_stream["url"] == _STREAM.url
        # The very box bash_exec runs in: same owner, same volume, found the
        # same way, but not counted as a turn.
        kwargs = box.get_box.await_args.kwargs
        assert box.get_box.await_args.args[0].key() == (
            f"copilot:e2b:sandbox:{session.session_id}"
        )
        assert kwargs["volume_mounts"] == {WORKSPACE_PATH: user_volume_name(_USER)}
        assert kwargs["template"] == "agpt-desktop-1x2"
        assert kwargs["count_turn"] is False
        assert kwargs["user_id"] == _USER
        assert kwargs["session_id"] == session.session_id
        box.desktop.ensure_display.assert_awaited_once()
        box.desktop.ensure_persistent_home.assert_awaited_once()
        # The screen flag is remembered against this box's id.
        assert box.redis.set.await_args.args[:2] == (
            f"copilot:e2b:sandbox:{session.session_id}:display",
            _BOX,
        )
        assert "Screen is on" in result.message
        assert "same machine" in result.message
        assert WORKSPACE_PATH in result.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_screen_already_on_just_refreshes_the_stream(self):
        tool = StartDesktopTool()
        session = make_session(user_id=_USER)
        box = _Box(display=_BOX)

        result = await _run(tool, box, user_id=_USER, session=session)

        assert isinstance(result, DesktopStreamToolResponse)
        assert "already on" in result.message
        box.desktop.ensure_display.assert_awaited_once()
        box.desktop.ensure_persistent_home.assert_not_awaited()
        box.desktop.start_stream.assert_awaited_once()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_no_user_id_means_no_volume_and_says_so(self):
        tool = StartDesktopTool()
        session = make_session(user_id=_USER)
        box = _Box(mounted=False)

        result = await _run(tool, box, user_id=None, session=session)

        assert isinstance(result, DesktopStreamToolResponse)
        assert box.get_box.await_args.kwargs["volume_mounts"] == {}
        assert "persists" in result.message
        box.desktop.ensure_persistent_home.assert_not_awaited()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_failure_is_reported_not_raised(self):
        tool = StartDesktopTool()
        session = make_session(user_id=_USER)
        box = _Box()
        box.get_box.side_effect = RuntimeError("e2b down")

        result = await _run(tool, box, user_id=_USER, session=session)

        assert isinstance(result, ErrorResponse)
        assert result.error == "desktop_start_failed"


class TestExpertDesktop:
    """An expert session's screen goes on in the expert's own computer."""

    _EXPERT = "exp-desktop-1"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_expert_screen_is_keyed_by_expert_with_home_and_shared(self):
        tool = StartDesktopTool()
        session = make_session(user_id=_USER, expert_id=self._EXPERT)
        box = _Box()

        result = await _run(tool, box, user_id=_USER, session=session)

        assert isinstance(result, DesktopStreamToolResponse)
        owner = box.get_box.await_args.args[0]
        assert owner.kind == "expert" and owner.id == self._EXPERT
        # Own home first, then the user's shared workspace.
        assert box.get_box.await_args.kwargs["volume_mounts"] == {
            WORKSPACE_PATH: expert_volume_name(self._EXPERT),
            SHARED_PATH: user_volume_name(_USER),
        }
        # Remembered under the expert, not the session: every session of this
        # expert must come back to the same screen.
        assert box.redis.set.await_args.args[0] == (
            f"copilot:e2b:expert:{self._EXPERT}:shell:display"
        )
        assert session.session_id not in box.redis.set.await_args.args[0]
        assert "own persistent computer" in result.message
        assert SHARED_PATH in result.message
