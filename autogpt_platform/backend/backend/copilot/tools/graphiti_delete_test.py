"""Tests for MemoryDeleteTool."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.model import ChatSession
from backend.copilot.tools.graphiti_delete import MemoryDeleteTool
from backend.copilot.tools.models import ErrorResponse, MemoryDeleteResponse


def _make_session(session_id: str = "test-session") -> ChatSession:
    return ChatSession(
        session_id=session_id,
        user_id="test-user",
        title=None,
        messages=[],
        usage=[],
        credentials={},
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


class TestMemoryDeleteTool:
    """Tests for MemoryDeleteTool._execute."""

    @pytest.mark.asyncio
    async def test_delete_no_user_returns_error(self):
        tool = MemoryDeleteTool()
        session = _make_session()

        result = await tool._execute(
            user_id=None,
            session=session,
            confirm=True,
        )

        assert isinstance(result, ErrorResponse)
        assert "Authentication required" in result.message
        assert result.session_id == "test-session"

    @pytest.mark.asyncio
    async def test_delete_feature_disabled_returns_error(self):
        tool = MemoryDeleteTool()
        session = _make_session()

        with patch(
            "backend.copilot.tools.graphiti_delete.is_enabled_for_user",
            new_callable=AsyncMock,
            return_value=False,
        ):
            result = await tool._execute(
                user_id="user-1",
                session=session,
                confirm=True,
            )

        assert isinstance(result, ErrorResponse)
        assert "not enabled" in result.message

    @pytest.mark.asyncio
    async def test_delete_confirm_false_returns_error(self):
        tool = MemoryDeleteTool()
        session = _make_session()

        with patch(
            "backend.copilot.tools.graphiti_delete.is_enabled_for_user",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await tool._execute(
                user_id="user-1",
                session=session,
                confirm=False,
            )

        assert isinstance(result, ErrorResponse)
        assert "confirm=true" in result.message
        assert "irreversible" in result.message

    @pytest.mark.asyncio
    async def test_delete_confirm_default_false_returns_error(self):
        """Calling without explicit confirm kwarg uses the default (False)."""
        tool = MemoryDeleteTool()
        session = _make_session()

        with patch(
            "backend.copilot.tools.graphiti_delete.is_enabled_for_user",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await tool._execute(
                user_id="user-1",
                session=session,
            )

        assert isinstance(result, ErrorResponse)
        assert "confirm=true" in result.message

    @pytest.mark.asyncio
    async def test_delete_success_clears_data_and_evicts_client(self):
        tool = MemoryDeleteTool()
        session = _make_session()

        mock_client = MagicMock()
        mock_client.driver = MagicMock()
        mock_clear_data = AsyncMock()
        mock_evict = MagicMock()

        with (
            patch(
                "backend.copilot.tools.graphiti_delete.is_enabled_for_user",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "backend.copilot.tools.graphiti_delete.derive_group_id",
                return_value="group-user-1",
            ),
            patch(
                "backend.copilot.tools.graphiti_delete.get_graphiti_client",
                new_callable=AsyncMock,
                return_value=mock_client,
            ),
            patch(
                "backend.copilot.tools.graphiti_delete.clear_data",
                mock_clear_data,
            ),
            patch(
                "backend.copilot.tools.graphiti_delete.evict_client",
                mock_evict,
            ),
        ):
            result = await tool._execute(
                user_id="user-1",
                session=session,
                confirm=True,
            )

        assert isinstance(result, MemoryDeleteResponse)
        assert "permanently deleted" in result.message
        assert result.session_id == "test-session"

        mock_clear_data.assert_awaited_once_with(
            mock_client.driver,
            group_ids=["group-user-1"],
        )
        mock_evict.assert_called_once_with("group-user-1")

    @pytest.mark.asyncio
    async def test_delete_success_derives_group_id_from_user(self):
        """Verify derive_group_id is called with the correct user_id."""
        tool = MemoryDeleteTool()
        session = _make_session()

        mock_client = MagicMock()
        mock_client.driver = MagicMock()
        mock_derive = MagicMock(return_value="group-abc")

        with (
            patch(
                "backend.copilot.tools.graphiti_delete.is_enabled_for_user",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "backend.copilot.tools.graphiti_delete.derive_group_id",
                mock_derive,
            ),
            patch(
                "backend.copilot.tools.graphiti_delete.get_graphiti_client",
                new_callable=AsyncMock,
                return_value=mock_client,
            ),
            patch(
                "backend.copilot.tools.graphiti_delete.clear_data",
                AsyncMock(),
            ),
            patch(
                "backend.copilot.tools.graphiti_delete.evict_client",
                MagicMock(),
            ),
        ):
            result = await tool._execute(
                user_id="abc-user-id",
                session=session,
                confirm=True,
            )

        assert isinstance(result, MemoryDeleteResponse)
        mock_derive.assert_called_once_with("abc-user-id")
