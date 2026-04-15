"""Tests for MemoryStoreTool."""

import json
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

        mock_enqueue.assert_awaited_once()
        call_kwargs = mock_enqueue.await_args.kwargs
        assert call_kwargs["name"] == "user_prefers_python"
        assert call_kwargs["source_description"] == "Direct statement"
        assert call_kwargs["is_json"] is True
        envelope = json.loads(call_kwargs["episode_body"])
        assert envelope["content"] == "The user prefers Python over JavaScript."
        assert envelope["memory_kind"] == "fact"

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
        mock_enqueue.assert_awaited_once()
        call_kwargs = mock_enqueue.await_args.kwargs
        assert call_kwargs["name"] == "some_fact"
        assert call_kwargs["source_description"] == "Conversation memory"
        assert call_kwargs["is_json"] is True
        envelope = json.loads(call_kwargs["episode_body"])
        assert envelope["content"] == "A fact worth remembering."

    @pytest.mark.asyncio
    async def test_store_invalid_source_kind_falls_back(self):
        """Invalid enum values should fall back to defaults, not crash."""
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
                content="A fact.",
                source_kind="INVALID_SOURCE",
                memory_kind="INVALID_KIND",
            )

        assert isinstance(result, MemoryStoreResponse)
        envelope = json.loads(mock_enqueue.await_args.kwargs["episode_body"])
        assert envelope["source_kind"] == "user_asserted"
        assert envelope["memory_kind"] == "fact"

    @pytest.mark.asyncio
    async def test_store_valid_enum_values_preserved(self):
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
                name="rule_1",
                content="Always CC Sarah.",
                source_kind="user_asserted",
                memory_kind="rule",
            )

        assert isinstance(result, MemoryStoreResponse)
        envelope = json.loads(mock_enqueue.await_args.kwargs["episode_body"])
        assert envelope["source_kind"] == "user_asserted"
        assert envelope["memory_kind"] == "rule"

    @pytest.mark.asyncio
    async def test_store_queue_full_returns_error(self):
        tool = MemoryStoreTool()
        session = _make_session()

        with (
            patch(
                "backend.copilot.tools.graphiti_store.is_enabled_for_user",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "backend.copilot.tools.graphiti_store.enqueue_episode",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            result = await tool._execute(
                user_id="user-1",
                session=session,
                name="pref",
                content="likes python",
            )

        assert isinstance(result, ErrorResponse)
        assert "queue" in result.message.lower()

    @pytest.mark.asyncio
    async def test_store_with_scope(self):
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
                name="project_note",
                content="CRM uses PostgreSQL.",
                scope="project:crm",
            )

        assert isinstance(result, MemoryStoreResponse)
        envelope = json.loads(mock_enqueue.await_args.kwargs["episode_body"])
        assert envelope["scope"] == "project:crm"
