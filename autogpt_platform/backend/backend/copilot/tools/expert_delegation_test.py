"""``has_chat_with``: the one read that tells an expert's first turn from a
later one."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from .expert_delegation import has_chat_with


def _chats(monkeypatch, **behaviour) -> MagicMock:
    chats = MagicMock()
    chats.get_user_chat_sessions = AsyncMock(**behaviour)
    monkeypatch.setattr(
        "backend.copilot.tools.expert_delegation.chat_db", lambda: chats
    )
    return chats


@pytest.mark.asyncio
async def test_a_chat_with_the_expert_counts(monkeypatch):
    _chats(monkeypatch, return_value=[MagicMock()])
    assert await has_chat_with("alice", "expert-b") is True


@pytest.mark.asyncio
async def test_no_chat_with_the_expert_is_a_first_turn(monkeypatch):
    _chats(monkeypatch, return_value=[])
    assert await has_chat_with("alice", "expert-b") is False


@pytest.mark.asyncio
async def test_it_lists_one_chat_and_counts_none(monkeypatch):
    """Only whether one exists: no total count is gathered beside it."""
    chats = _chats(monkeypatch, return_value=[])
    await has_chat_with("alice", "expert-b")
    chats.get_user_chat_sessions.assert_awaited_once_with(
        "alice", limit=1, expert_id="expert-b", pinned_first=False
    )
    chats.get_user_session_count.assert_not_called()


@pytest.mark.asyncio
async def test_a_failed_read_counts_as_a_chat(monkeypatch):
    """A missed hello is the safer mistake than telling the expert a chat
    already under way is their first."""
    _chats(monkeypatch, side_effect=RuntimeError("db down"))
    assert await has_chat_with("alice", "expert-b") is True
