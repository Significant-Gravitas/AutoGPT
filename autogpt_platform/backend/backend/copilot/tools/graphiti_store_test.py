"""Tests for MemoryStoreTool."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.model import ChatSession
from backend.copilot.tools.graphiti_store import MemoryStoreTool
from backend.copilot.tools.models import ErrorResponse, MemoryStoreResponse


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


class TestMemoryStoreTool:
    """Tests for MemoryStoreTool._execute."""

    @pytest.mark.asyncio
    async def test_store_no_user_returns_error(self):
        tool = MemoryStoreTool()
        session = _make_session()

        result = await tool._execute(
            user_id=None,
            session=session,
            name="pref",
            content="likes python",
        )

        assert isinstance(result, ErrorResponse)
        assert "Authentication required" in result.message
        assert result.session_id == "test-session"

    @pytest.mark.asyncio
    async def test_store_feature_disabled_returns_error(self):
        tool = MemoryStoreTool()
        session = _make_session()

        with patch(
            "backend.copilot.tools.graphiti_store.is_enabled_for_user",
            new_callable=AsyncMock,
            return_value=False,
        ):
            result = await tool._execute(
                user_id="user-1",
                session=session,
                name="pref",
                content="likes python",
            )

        assert isinstance(result, ErrorResponse)
        assert "not enabled" in result.message

    @pytest.mark.asyncio
    async def test_store_missing_name_returns_error(self):
        tool = MemoryStoreTool()
        session = _make_session()

        with patch(
            "backend.copilot.tools.graphiti_store.is_enabled_for_user",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await tool._execute(
                user_id="user-1",
                session=session,
                name="",
                content="likes python",
            )

        assert isinstance(result, ErrorResponse)
        assert "'name' and 'content' are required" in result.message

    @pytest.mark.asyncio
    async def test_store_missing_content_returns_error(self):
        tool = MemoryStoreTool()
        session = _make_session()

        with patch(
            "backend.copilot.tools.graphiti_store.is_enabled_for_user",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await tool._execute(
                user_id="user-1",
                session=session,
                name="pref",
                content="",
            )

        assert isinstance(result, ErrorResponse)
        assert "'name' and 'content' are required" in result.message

    @pytest.mark.asyncio
    async def test_store_missing_both_name_and_content_returns_error(self):
        tool = MemoryStoreTool()
        session = _make_session()

        with patch(
            "backend.copilot.tools.graphiti_store.is_enabled_for_user",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await tool._execute(
                user_id="user-1",
                session=session,
                name="",
                content="",
            )

        assert isinstance(result, ErrorResponse)
        assert "'name' and 'content' are required" in result.message

    @pytest.mark.asyncio
    async def test_store_success_enqueues_episode(self):
        tool = MemoryStoreTool()
        session = _make_session()

        mock_enqueue = AsyncMock()

        with (
            patch(
                "backend.copilot.tools.graphiti_store.is_enabled_for_user",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "backend.copilot.tools.graphiti_store.enqueue_episode",
                mock_enqueue,
            ),
        ):
            result = await tool._execute(
                user_id="user-1",
                session=session,
                name="user_prefers_python",
                content="The user prefers Python over JavaScript.",
                source_description="Direct statement",
            )

        assert isinstance(result, MemoryStoreResponse)
        assert result.memory_name == "user_prefers_python"
        assert "queued for storage" in result.message
        assert result.session_id == "test-session"

        mock_enqueue.assert_awaited_once_with(
            "user-1",
            "test-session",
            name="user_prefers_python",
            episode_body="The user prefers Python over JavaScript.",
            source_description="Direct statement",
        )

    @pytest.mark.asyncio
    async def test_store_success_uses_default_source_description(self):
        tool = MemoryStoreTool()
        session = _make_session()

        mock_enqueue = AsyncMock()

        with (
            patch(
                "backend.copilot.tools.graphiti_store.is_enabled_for_user",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "backend.copilot.tools.graphiti_store.enqueue_episode",
                mock_enqueue,
            ),
        ):
            result = await tool._execute(
                user_id="user-1",
                session=session,
                name="some_fact",
                content="A fact worth remembering.",
            )

        assert isinstance(result, MemoryStoreResponse)
        mock_enqueue.assert_awaited_once_with(
            "user-1",
            "test-session",
            name="some_fact",
            episode_body="A fact worth remembering.",
            source_description="Conversation memory",
        )
