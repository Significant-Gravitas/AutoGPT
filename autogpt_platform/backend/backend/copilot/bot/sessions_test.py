"""Tests for the per-target copilot session cache helpers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.bot import sessions, threads
from backend.copilot.bot.config import SESSION_TTL


def test_session_ttl_outlives_thread_subscription():
    # If the session cache expires before the thread stops auto-replying, a
    # follow-up message silently starts a fresh copilot session and loses the
    # whole conversation. The session must live at least as long as the bot
    # keeps replying in the thread.
    assert SESSION_TTL >= threads.THREAD_SUBSCRIPTION_TTL


def test_session_cache_key_format():
    assert (
        sessions.session_cache_key("discord", "123")
        == "copilot-bot:session:discord:123"
    )


@pytest.mark.asyncio
async def test_clear_session_deletes_key():
    redis = MagicMock()
    redis.delete = AsyncMock()
    with patch(
        "backend.copilot.bot.sessions.get_redis_async",
        new=AsyncMock(return_value=redis),
    ):
        await sessions.clear_session("discord", "123")

    redis.delete.assert_awaited_once_with("copilot-bot:session:discord:123")


@pytest.mark.asyncio
async def test_set_session_writes_with_ttl():
    redis = MagicMock()
    redis.set = AsyncMock()
    with patch(
        "backend.copilot.bot.sessions.get_redis_async",
        new=AsyncMock(return_value=redis),
    ):
        await sessions.set_session("discord", "123", "sess-9")

    redis.set.assert_awaited_once_with(
        "copilot-bot:session:discord:123", "sess-9", ex=sessions.SESSION_TTL
    )
