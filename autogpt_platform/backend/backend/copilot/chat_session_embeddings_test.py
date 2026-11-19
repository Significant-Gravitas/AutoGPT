"""Tests for the chat-session title embedding scheduler."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import ContentType
from prisma.errors import ClientNotConnectedError

from backend.copilot.chat_session_embeddings import (
    _run_embedding,
    delete_chat_session_embedding,
    schedule_chat_session_embedding,
)

_MODULE = "backend.copilot.chat_session_embeddings"


@pytest.fixture
def db(mocker):
    db = MagicMock()
    db.get_content_embedding = AsyncMock(return_value=None)
    db.store_content_embedding = AsyncMock(return_value=True)
    db.delete_content_embedding = AsyncMock(return_value=True)
    mocker.patch(f"{_MODULE}.embeddings_db", return_value=db)
    mocker.patch(f"{_MODULE}.generate_embedding", AsyncMock(return_value=[0.5]))
    return db


@pytest.mark.asyncio
async def test_run_embedding_deletes_when_title_is_empty(db):
    """An empty/whitespace title drops any stale row instead of embedding."""
    await _run_embedding("sess-1", "user-1", "   ")
    db.store_content_embedding.assert_not_called()
    db.delete_content_embedding.assert_awaited_once_with(
        ContentType.CHAT_SESSION, "sess-1", user_id="user-1"
    )


@pytest.mark.asyncio
async def test_run_embedding_stores_the_title_for_the_user(db):
    await _run_embedding("sess-1", "user-1", "My Chat")
    kwargs = db.store_content_embedding.call_args.kwargs
    assert kwargs["content_type"] == ContentType.CHAT_SESSION
    assert kwargs["content_id"] == "sess-1"
    assert kwargs["user_id"] == "user-1"
    assert kwargs["embedding"] == [0.5]
    assert kwargs["searchable_text"] == "My Chat"
    assert kwargs["metadata"] == {"title": "My Chat"}


@pytest.mark.asyncio
async def test_run_embedding_skips_when_title_unchanged(db):
    """If the existing row already carries the same title, skip re-embedding."""
    db.get_content_embedding.return_value = {"searchableText": "My Chat"}
    await _run_embedding("sess-1", "user-1", "My Chat")
    db.store_content_embedding.assert_not_called()


@pytest.mark.asyncio
async def test_run_embedding_refreshes_when_title_changed(db):
    db.get_content_embedding.return_value = {"searchableText": "Old Title"}
    await _run_embedding("sess-1", "user-1", "New Title")
    db.store_content_embedding.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_embedding_swallows_failures(db):
    """Failures must not propagate — it's a best-effort background task."""
    db.store_content_embedding.side_effect = RuntimeError("openai down")
    # Must not raise.
    await _run_embedding("sess-1", "user-1", "My Chat")


@pytest.mark.asyncio
async def test_schedule_returns_task_and_runs_in_background(db):
    task = schedule_chat_session_embedding("sess-1", "user-1", "My Chat")
    assert isinstance(task, asyncio.Task)
    await task
    db.store_content_embedding.assert_awaited_once()


@pytest.mark.asyncio
async def test_schedule_handles_none_title(db):
    """A ``None`` title is coerced to empty and triggers the delete path."""
    task = schedule_chat_session_embedding("sess-1", "user-1", None)
    await task
    db.delete_content_embedding.assert_awaited_once_with(
        ContentType.CHAT_SESSION, "sess-1", user_id="user-1"
    )


@pytest.mark.asyncio
async def test_delete_chat_session_embedding_forwards_user_scope(db):
    await delete_chat_session_embedding("sess-1", "user-1")
    db.delete_content_embedding.assert_awaited_once_with(
        ContentType.CHAT_SESSION, "sess-1", user_id="user-1"
    )


@pytest.mark.asyncio
async def test_delete_chat_session_embedding_swallows_failures(db):
    """Cleanup failures must not propagate."""
    db.delete_content_embedding.side_effect = RuntimeError("db down")
    # Must not raise.
    await delete_chat_session_embedding("sess-1", "user-1")


@pytest.fixture
def copilot_executor_db(mocker):
    """Prisma disconnected in-process, the DatabaseManager client answering.

    Titles are set from the copilot executor, which never connects Prisma.
    """
    mocker.patch("backend.data.db.is_connected", return_value=False)
    for raw in ("query_raw_with_schema", "execute_raw_with_schema"):
        mocker.patch(
            f"backend.api.features.search.embeddings.{raw}",
            new=AsyncMock(side_effect=ClientNotConnectedError()),
        )
    openai = MagicMock()
    openai.embeddings.create = AsyncMock(
        return_value=MagicMock(data=[MagicMock(embedding=[0.5, 0.25])])
    )
    mocker.patch(
        "backend.api.features.search.embeddings.get_openai_client",
        return_value=openai,
    )

    db_manager = MagicMock()
    db_manager.get_content_embedding = AsyncMock(return_value=None)
    db_manager.store_content_embedding = AsyncMock(return_value=True)
    db_manager.delete_content_embedding = AsyncMock(return_value=True)
    mocker.patch(
        "backend.util.clients.get_database_manager_async_client",
        return_value=db_manager,
    )
    return db_manager


@pytest.mark.asyncio
async def test_title_embedding_is_stored_without_prisma(copilot_executor_db, caplog):
    await _run_embedding("sess-1", "user-1", "My Chat")

    assert "Failed to ensure chat session embedding" not in caplog.text
    copilot_executor_db.get_content_embedding.assert_awaited_once_with(
        ContentType.CHAT_SESSION, "sess-1", "user-1"
    )
    kwargs = copilot_executor_db.store_content_embedding.await_args.kwargs
    assert kwargs["content_type"] == ContentType.CHAT_SESSION
    assert kwargs["content_id"] == "sess-1"
    assert kwargs["user_id"] == "user-1"
    assert kwargs["embedding"] == [0.5, 0.25]
    assert kwargs["searchable_text"] == "My Chat"


@pytest.mark.asyncio
async def test_title_embedding_is_deleted_without_prisma(copilot_executor_db, caplog):
    await delete_chat_session_embedding("sess-1", "user-1")

    assert "Failed to delete" not in caplog.text
    copilot_executor_db.delete_content_embedding.assert_awaited_once_with(
        ContentType.CHAT_SESSION, "sess-1", user_id="user-1"
    )
