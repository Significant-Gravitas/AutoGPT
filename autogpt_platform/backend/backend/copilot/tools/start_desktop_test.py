"""Tests for StartDesktopTool — session-linked on-demand E2B desktop."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.desktop._api import (
    SHARED_PATH,
    WORKSPACE_PATH,
    DesktopStream,
    PersistenceInfo,
)
from backend.blocks.desktop._common import expert_volume_name, user_volume_name
from backend.util.sandbox_metadata import deployment_env

from ._test_data import make_session
from .models import DesktopStreamToolResponse, ErrorResponse
from .start_desktop import StartDesktopTool

_USER = "user-start-desktop-test"
_STREAM = DesktopStream(
    url="https://6080-sbx.e2b.app/vnc.html?autoconnect=true",
    sandbox_id="sbx-desktop-1",
)


def _make_redis(stored: str | None = None) -> MagicMock:
    redis = MagicMock()
    redis.get = AsyncMock(return_value=stored)
    redis.set = AsyncMock()
    redis.delete = AsyncMock()
    return redis


def _make_desktop() -> MagicMock:
    desktop = MagicMock()
    desktop.sandbox_id = "sbx-desktop-1"
    desktop.ensure_display = AsyncMock()
    desktop.start_stream = AsyncMock(return_value=_STREAM)
    desktop.is_workspace_mounted = AsyncMock(return_value=True)
    return desktop


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
    async def test_creates_desktop_on_shared_user_volume(self):
        tool = StartDesktopTool()
        session = make_session(user_id=_USER)
        desktop = _make_desktop()
        redis = _make_redis(stored=None)

        with (
            patch(
                "backend.copilot.computer.get_redis_async",
                new=AsyncMock(return_value=redis),
            ),
            patch("backend.copilot.computer.DesktopSession") as mock_session_cls,
            patch("backend.copilot.tools.start_desktop.chat_config") as mock_config,
        ):
            mock_config.active_e2b_api_key = "e2b_test_key"
            mock_config.e2b_desktop_timeout = 900
            mock_config.e2b_desktop_template = "desktop"
            mock_session_cls.create = AsyncMock(
                return_value=(desktop, PersistenceInfo(volume_mounted=True))
            )

            result = await tool._execute(user_id=_USER, session=session)

        assert isinstance(result, DesktopStreamToolResponse)
        assert result.desktop_stream["kind"] == "desktop_stream"
        assert result.desktop_stream["url"] == _STREAM.url
        # The desktop mounts the SAME per-user volume as the agent shell, and
        # is tagged so its owner can be found through the E2B API.
        create_kwargs = mock_session_cls.create.await_args.kwargs
        assert create_kwargs["volume_mounts"] == {
            WORKSPACE_PATH: user_volume_name(_USER)
        }
        assert create_kwargs["metadata"] == {
            "autogpt_owner": f"session:{session.session_id}",
            "autogpt_kind": "desktop",
            "autogpt_source": "copilot",
            "autogpt_env": deployment_env(),
            "autogpt_user": _USER,
            "autogpt_session": session.session_id,
            "autogpt_template": "desktop",
            "autogpt_mounts": "attached",
        }
        redis.set.assert_awaited()
        assert redis.set.await_args.args[0] == (
            f"copilot:e2b:desktop:{session.session_id}"
        )
        assert "started" in result.message
        assert "workspace" in result.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_resumes_existing_desktop_without_creating(self):
        tool = StartDesktopTool()
        session = make_session(user_id=_USER)
        desktop = _make_desktop()
        redis = _make_redis(stored="sbx-desktop-1")

        with (
            patch(
                "backend.copilot.computer.get_redis_async",
                new=AsyncMock(return_value=redis),
            ),
            patch("backend.copilot.computer.DesktopSession") as mock_session_cls,
            patch("backend.copilot.tools.start_desktop.chat_config") as mock_config,
        ):
            mock_config.active_e2b_api_key = "e2b_test_key"
            mock_session_cls.connect = AsyncMock(return_value=desktop)
            mock_session_cls.create = AsyncMock()

            result = await tool._execute(user_id=_USER, session=session)

        assert isinstance(result, DesktopStreamToolResponse)
        mock_session_cls.connect.assert_awaited_once_with(
            "sbx-desktop-1", "e2b_test_key"
        )
        mock_session_cls.create.assert_not_awaited()
        desktop.ensure_display.assert_awaited_once()
        assert "resumed" in result.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_no_user_id_creates_ephemeral_desktop(self):
        tool = StartDesktopTool()
        session = make_session(user_id=_USER)
        desktop = _make_desktop()
        redis = _make_redis(stored=None)

        with (
            patch(
                "backend.copilot.computer.get_redis_async",
                new=AsyncMock(return_value=redis),
            ),
            patch("backend.copilot.computer.DesktopSession") as mock_session_cls,
            patch("backend.copilot.tools.start_desktop.chat_config") as mock_config,
        ):
            mock_config.active_e2b_api_key = "e2b_test_key"
            mock_config.e2b_desktop_timeout = 900
            mock_config.e2b_desktop_template = "desktop"
            mock_session_cls.create = AsyncMock(
                return_value=(desktop, PersistenceInfo(volume_mounted=False))
            )

            result = await tool._execute(user_id=None, session=session)

        assert isinstance(result, DesktopStreamToolResponse)
        assert mock_session_cls.create.await_args.kwargs["volume_mounts"] is None
        assert "ephemeral" in result.message


class TestExpertDesktop:
    """An expert session's desktop is the expert's own persistent computer."""

    _EXPERT = "exp-desktop-1"

    def _patches(self, redis, mock_session_cls, mock_config, found: str | None):
        mock_config.active_e2b_api_key = "e2b_test_key"
        mock_config.e2b_desktop_timeout = 900
        mock_config.e2b_desktop_template = "desktop"
        return (
            patch(
                "backend.copilot.computer.get_redis_async",
                new=AsyncMock(return_value=redis),
            ),
            patch(
                "backend.copilot.computer.find_owned_sandbox_id",
                new=AsyncMock(return_value=found),
            ),
        )

    @pytest.mark.asyncio(loop_scope="session")
    async def test_expert_desktop_is_keyed_by_expert_with_home_and_shared(self):
        tool = StartDesktopTool()
        session = make_session(user_id=_USER, expert_id=self._EXPERT)
        desktop = _make_desktop()
        redis = _make_redis(stored=None)

        with (
            patch("backend.copilot.computer.DesktopSession") as mock_session_cls,
            patch("backend.copilot.tools.start_desktop.chat_config") as mock_config,
        ):
            redis_patch, find_patch = self._patches(
                redis, mock_session_cls, mock_config, found=None
            )
            mock_session_cls.create = AsyncMock(
                return_value=(
                    desktop,
                    PersistenceInfo(
                        volume_mounted=True,
                        mounted_paths=[WORKSPACE_PATH, SHARED_PATH],
                    ),
                )
            )
            with redis_patch, find_patch:
                result = await tool._execute(user_id=_USER, session=session)

        assert isinstance(result, DesktopStreamToolResponse)
        create_kwargs = mock_session_cls.create.await_args.kwargs
        # Own home first, then the user's shared workspace.
        assert create_kwargs["volume_mounts"] == {
            WORKSPACE_PATH: expert_volume_name(self._EXPERT),
            SHARED_PATH: user_volume_name(_USER),
        }
        assert create_kwargs["metadata"] == {
            "autogpt_owner": f"expert:{self._EXPERT}",
            "autogpt_kind": "desktop",
            "autogpt_source": "copilot",
            "autogpt_env": deployment_env(),
            "autogpt_user": _USER,
            "autogpt_session": session.session_id,
            "autogpt_expert": self._EXPERT,
            "autogpt_template": "desktop",
            "autogpt_mounts": "attached",
        }
        # Cached under the expert, not the session: every session of this
        # expert must come back to the same desktop.
        assert redis.set.await_args.args[0] == (
            f"copilot:e2b:expert:{self._EXPERT}:desktop"
        )
        assert session.session_id not in redis.set.await_args.args[0]
        assert "own persistent computer" in result.message
        assert SHARED_PATH in result.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_expert_desktop_recovered_from_e2b_when_cache_is_empty(self):
        tool = StartDesktopTool()
        session = make_session(user_id=_USER, expert_id=self._EXPERT)
        desktop = _make_desktop()
        redis = _make_redis(stored=None)

        with (
            patch("backend.copilot.computer.DesktopSession") as mock_session_cls,
            patch("backend.copilot.tools.start_desktop.chat_config") as mock_config,
        ):
            redis_patch, find_patch = self._patches(
                redis, mock_session_cls, mock_config, found="sbx-desktop-1"
            )
            mock_session_cls.connect = AsyncMock(return_value=desktop)
            mock_session_cls.create = AsyncMock()
            with redis_patch, find_patch:
                result = await tool._execute(user_id=_USER, session=session)

        assert isinstance(result, DesktopStreamToolResponse)
        mock_session_cls.connect.assert_awaited_once_with(
            "sbx-desktop-1", "e2b_test_key"
        )
        mock_session_cls.create.assert_not_awaited()
        # The recovered id is re-cached under the expert key.
        assert redis.set.await_args.args[0] == (
            f"copilot:e2b:expert:{self._EXPERT}:desktop"
        )
        assert "resumed" in result.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_plain_session_never_asks_e2b_for_a_lost_desktop(self):
        tool = StartDesktopTool()
        session = make_session(user_id=_USER)
        desktop = _make_desktop()
        redis = _make_redis(stored=None)

        with (
            patch(
                "backend.copilot.computer.get_redis_async",
                new=AsyncMock(return_value=redis),
            ),
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_sandbox,
            patch("backend.copilot.computer.DesktopSession") as mock_session_cls,
            patch("backend.copilot.tools.start_desktop.chat_config") as mock_config,
        ):
            mock_config.active_e2b_api_key = "e2b_test_key"
            mock_config.e2b_desktop_timeout = 900
            mock_config.e2b_desktop_template = "desktop"
            mock_session_cls.create = AsyncMock(
                return_value=(desktop, PersistenceInfo(volume_mounted=True))
            )
            result = await tool._execute(user_id=_USER, session=session)

        assert isinstance(result, DesktopStreamToolResponse)
        mock_sandbox.list.assert_not_called()
