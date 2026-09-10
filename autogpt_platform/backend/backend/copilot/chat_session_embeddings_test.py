"""Tests for the chat-session title embedding scheduler."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import ContentType

from backend.copilot.chat_session_embeddings import (
    _run_embedding,
    delete_chat_session_embedding,
    schedule_chat_session_embedding,
)


@pytest.mark.asyncio
async def test_run_embedding_deletes_when_title_is_empty():
    """An empty/whitespace title drops any stale row instead of embedding."""
    with (
        patch(
            "backend.copilot.chat_session_embeddings.delete_content_embedding",
            new=AsyncMock(return_value=None),
        ) as mock_delete,
        patch(
            "backend.copilot.chat_session_embeddings.ensure_content_embedding",
            new=AsyncMock(return_value=True),
        ) as mock_ensure,
    ):
        await _run_embedding("sess-1", "user-1", "   ")
    mock_ensure.assert_not_called()
    mock_delete.assert_awaited_once_with(
        ContentType.CHAT_SESSION, "sess-1", user_id="user-1"
    )


@pytest.mark.asyncio
async def test_run_embedding_forwards_user_scope_and_force_true():
    with (
        patch(
            "backend.copilot.chat_session_embeddings.get_content_embedding",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "backend.copilot.chat_session_embeddings.ensure_content_embedding",
            new=AsyncMock(return_value=True),
        ) as mock_ensure,
    ):
        await _run_embedding("sess-1", "user-1", "My Chat")
    kwargs = mock_ensure.call_args.kwargs
    assert kwargs["content_type"] == ContentType.CHAT_SESSION
    assert kwargs["content_id"] == "sess-1"
    assert kwargs["user_id"] == "user-1"
    assert kwargs["searchable_text"] == "My Chat"
    assert kwargs["metadata"] == {"title": "My Chat"}
    assert kwargs["force"] is True


@pytest.mark.asyncio
async def test_run_embedding_skips_when_title_unchanged():
    """If the existing row already carries the same title, skip re-embedding."""
    with (
        patch(
            "backend.copilot.chat_session_embeddings.get_content_embedding",
            new=AsyncMock(return_value={"searchableText": "My Chat"}),
        ),
        patch(
            "backend.copilot.chat_session_embeddings.ensure_content_embedding",
            new=AsyncMock(return_value=True),
        ) as mock_ensure,
    ):
        await _run_embedding("sess-1", "user-1", "My Chat")
    mock_ensure.assert_not_called()


@pytest.mark.asyncio
async def test_run_embedding_refreshes_when_title_changed():
    with (
        patch(
            "backend.copilot.chat_session_embeddings.get_content_embedding",
            new=AsyncMock(return_value={"searchableText": "Old Title"}),
        ),
        patch(
            "backend.copilot.chat_session_embeddings.ensure_content_embedding",
            new=AsyncMock(return_value=True),
        ) as mock_ensure,
    ):
        await _run_embedding("sess-1", "user-1", "New Title")
    mock_ensure.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_embedding_swallows_failures():
    """Failures must not propagate — it's a best-effort background task."""
    with (
        patch(
            "backend.copilot.chat_session_embeddings.get_content_embedding",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "backend.copilot.chat_session_embeddings.ensure_content_embedding",
            new=AsyncMock(side_effect=RuntimeError("openai down")),
        ),
    ):
        # Must not raise.
        await _run_embedding("sess-1", "user-1", "My Chat")


@pytest.mark.asyncio
async def test_schedule_returns_task_and_runs_in_background():
    with (
        patch(
            "backend.copilot.chat_session_embeddings.get_content_embedding",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "backend.copilot.chat_session_embeddings.ensure_content_embedding",
            new=AsyncMock(return_value=True),
        ) as mock_ensure,
    ):
        task = schedule_chat_session_embedding("sess-1", "user-1", "My Chat")
        assert isinstance(task, asyncio.Task)
        await task
    mock_ensure.assert_awaited_once()


@pytest.mark.asyncio
async def test_schedule_handles_none_title():
    """A ``None`` title is coerced to empty and triggers the delete path."""
    with patch(
        "backend.copilot.chat_session_embeddings.delete_content_embedding",
        new=AsyncMock(return_value=None),
    ) as mock_delete:
        task = schedule_chat_session_embedding("sess-1", "user-1", None)
        await task
    mock_delete.assert_awaited_once_with(
        ContentType.CHAT_SESSION, "sess-1", user_id="user-1"
    )


@pytest.mark.asyncio
async def test_delete_chat_session_embedding_forwards_user_scope():
    with patch(
        "backend.copilot.chat_session_embeddings.delete_content_embedding",
        new=AsyncMock(return_value=None),
    ) as mock_delete:
        await delete_chat_session_embedding("sess-1", "user-1")
    mock_delete.assert_awaited_once_with(
        ContentType.CHAT_SESSION, "sess-1", user_id="user-1"
    )


@pytest.mark.asyncio
async def test_delete_chat_session_embedding_swallows_failures():
    """Cleanup failures must not propagate."""
    with patch(
        "backend.copilot.chat_session_embeddings.delete_content_embedding",
        new=AsyncMock(side_effect=RuntimeError("db down")),
    ):
        # Must not raise.
        await delete_chat_session_embedding("sess-1", "user-1")
