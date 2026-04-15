"""Unit tests for pending_message_helpers."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot import pending_message_helpers as helpers_module
from backend.copilot.pending_message_helpers import (
    PENDING_CALL_LIMIT,
    check_pending_call_rate,
    drain_pending_safe,
    insert_pending_before_last,
    persist_session_safe,
)
from backend.copilot.pending_messages import PendingMessage

# ── check_pending_call_rate ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_check_pending_call_rate_returns_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_redis = MagicMock()
    mock_redis.eval = AsyncMock(return_value=3)
    monkeypatch.setattr(
        helpers_module, "get_redis_async", AsyncMock(return_value=mock_redis)
    )

    result = await check_pending_call_rate("user-1")
    assert result == 3


@pytest.mark.asyncio
async def test_check_pending_call_rate_fails_open_on_redis_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        helpers_module,
        "get_redis_async",
        AsyncMock(side_effect=ConnectionError("down")),
    )

    result = await check_pending_call_rate("user-1")
    assert result == 0


@pytest.mark.asyncio
async def test_check_pending_call_rate_at_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_redis = MagicMock()
    mock_redis.eval = AsyncMock(return_value=PENDING_CALL_LIMIT + 1)
    monkeypatch.setattr(
        helpers_module, "get_redis_async", AsyncMock(return_value=mock_redis)
    )

    result = await check_pending_call_rate("user-1")
    assert result > PENDING_CALL_LIMIT


# ── drain_pending_safe ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_drain_pending_safe_returns_content_strings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    msgs = [PendingMessage(content="hello"), PendingMessage(content="world")]
    monkeypatch.setattr(
        helpers_module, "drain_pending_messages", AsyncMock(return_value=msgs)
    )

    result = await drain_pending_safe("sess-1")
    assert result == ["hello", "world"]


@pytest.mark.asyncio
async def test_drain_pending_safe_returns_empty_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        helpers_module,
        "drain_pending_messages",
        AsyncMock(side_effect=RuntimeError("redis down")),
    )

    result = await drain_pending_safe("sess-1", "[Test]")
    assert result == []


@pytest.mark.asyncio
async def test_drain_pending_safe_empty_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        helpers_module, "drain_pending_messages", AsyncMock(return_value=[])
    )

    result = await drain_pending_safe("sess-1")
    assert result == []


# ── insert_pending_before_last ─────────────────────────────────────────


def _make_session(*contents: str) -> Any:
    session = MagicMock()
    session.messages = [MagicMock(role="user", content=c) for c in contents]
    return session


def test_insert_pending_before_last_single_existing_message() -> None:
    session = _make_session("current")
    insert_pending_before_last(session, ["queued"])
    assert session.messages[0].content == "queued"
    assert session.messages[1].content == "current"


def test_insert_pending_before_last_multiple_pending() -> None:
    session = _make_session("current")
    insert_pending_before_last(session, ["p1", "p2"])
    contents = [m.content for m in session.messages]
    assert contents == ["p1", "p2", "current"]


def test_insert_pending_before_last_empty_session() -> None:
    session = _make_session()
    insert_pending_before_last(session, ["queued"])
    assert session.messages[0].content == "queued"


def test_insert_pending_before_last_no_texts_is_noop() -> None:
    session = _make_session("current")
    insert_pending_before_last(session, [])
    assert len(session.messages) == 1


# ── persist_session_safe ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_persist_session_safe_returns_updated_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = MagicMock()
    updated = MagicMock()
    monkeypatch.setattr(
        helpers_module, "upsert_chat_session", AsyncMock(return_value=updated)
    )

    result = await persist_session_safe(original, "[Test]")
    assert result is updated


@pytest.mark.asyncio
async def test_persist_session_safe_returns_original_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = MagicMock()
    monkeypatch.setattr(
        helpers_module,
        "upsert_chat_session",
        AsyncMock(side_effect=Exception("db error")),
    )

    result = await persist_session_safe(original, "[Test]")
    assert result is original
