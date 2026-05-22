"""Tests for the per-target copilot session cache helpers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.bot import sessions


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
