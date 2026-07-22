"""Tests for StartDesktopTool — session-linked on-demand E2B desktop."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.desktop._api import DesktopStream

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
    async def test_creates_desktop_and_returns_stream(self):
        tool = StartDesktopTool()
        session = make_session(user_id=_USER)
        desktop = _make_desktop()
        redis = _make_redis(stored=None)

        with (
            patch(
                "backend.copilot.tools.start_desktop.get_redis_async",
                new=AsyncMock(return_value=redis),
            ),
            patch(
                "backend.copilot.tools.start_desktop.DesktopSession"
            ) as mock_session_cls,
            patch(
                "backend.copilot.tools.start_desktop.get_current_sandbox",
                return_value=None,
            ),
            patch("backend.copilot.tools.start_desktop.chat_config") as mock_config,
        ):
            mock_config.active_e2b_api_key = "e2b_test_key"
            mock_config.e2b_desktop_timeout = 900
            mock_config.e2b_desktop_template = "desktop"
            mock_session_cls.create = AsyncMock(return_value=(desktop, MagicMock()))

            result = await tool._execute(user_id=_USER, session=session)

        assert isinstance(result, DesktopStreamToolResponse)
        assert result.desktop_stream["kind"] == "desktop_stream"
        assert result.desktop_stream["url"] == _STREAM.url
        mock_session_cls.create.assert_awaited_once()
        redis.set.assert_awaited()
        assert "started" in result.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_resumes_existing_desktop_without_creating(self):
        tool = StartDesktopTool()
        session = make_session(user_id=_USER)
        desktop = _make_desktop()
        redis = _make_redis(stored="sbx-desktop-1")

        with (
            patch(
                "backend.copilot.tools.start_desktop.get_redis_async",
                new=AsyncMock(return_value=redis),
            ),
            patch(
                "backend.copilot.tools.start_desktop.DesktopSession"
            ) as mock_session_cls,
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
