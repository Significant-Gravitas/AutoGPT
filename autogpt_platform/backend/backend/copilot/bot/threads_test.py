"""Tests for Redis-backed thread subscription tracking."""

from unittest.mock import AsyncMock, patch

import pytest

from . import threads


@pytest.fixture
def redis_mock():
    mock = AsyncMock()
    mock.get = AsyncMock()
    mock.set = AsyncMock()
    with patch("backend.copilot.bot.threads.get_redis_async", return_value=mock):
        yield mock


class TestSubscribe:
    @pytest.mark.asyncio
    async def test_writes_key_with_ttl(self, redis_mock):
        await threads.subscribe("discord", "thread-123")
        redis_mock.set.assert_awaited_once_with(
            "copilot-bot:thread:discord:thread-123",
            "1",
            ex=threads.THREAD_SUBSCRIPTION_TTL,
        )

    @pytest.mark.asyncio
    async def test_key_includes_platform(self, redis_mock):
        await threads.subscribe("telegram", "t-1")
        key = redis_mock.set.await_args.args[0]
        assert "telegram" in key
        assert "t-1" in key


class TestIsSubscribed:
    @pytest.mark.asyncio
    async def test_returns_true_when_present(self, redis_mock):
        redis_mock.get.return_value = "1"
        assert await threads.is_subscribed("discord", "thread-1") is True

    @pytest.mark.asyncio
    async def test_returns_false_when_missing(self, redis_mock):
        redis_mock.get.return_value = None
        assert await threads.is_subscribed("discord", "thread-1") is False

    @pytest.mark.asyncio
    async def test_uses_same_key_as_subscribe(self, redis_mock):
        redis_mock.get.return_value = None
        await threads.is_subscribed("discord", "thread-1")
        await threads.subscribe("discord", "thread-1")
        read_key = redis_mock.get.await_args.args[0]
        write_key = redis_mock.set.await_args.args[0]
        assert read_key == write_key
