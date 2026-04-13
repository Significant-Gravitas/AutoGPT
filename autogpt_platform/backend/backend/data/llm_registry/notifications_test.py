"""Tests for LLM registry pub/sub notifications (notifications.py).

Covers:
- publish_registry_refresh_notification: happy path and Redis error swallowed
- subscribe_to_registry_refresh: message triggers on_refresh, non-message
  types ignored, wrong channel ignored, CancelledError stops the loop,
  connection errors trigger reconnect
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data.llm_registry.notifications import (
    REGISTRY_REFRESH_CHANNEL,
    publish_registry_refresh_notification,
    subscribe_to_registry_refresh,
)


# ---------------------------------------------------------------------------
# publish_registry_refresh_notification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_sends_to_correct_channel(mocker):
    """publish_registry_refresh_notification publishes on the registry channel."""
    mock_redis = AsyncMock()
    mocker.patch(
        "backend.data.redis_client.get_redis_async",
        return_value=mock_redis,
    )

    await publish_registry_refresh_notification()

    mock_redis.publish.assert_called_once_with(REGISTRY_REFRESH_CHANNEL, "refresh")


@pytest.mark.asyncio
async def test_publish_swallows_redis_error(mocker):
    """Redis errors during publish are caught and logged, not raised."""
    mocker.patch(
        "backend.data.redis_client.get_redis_async",
        side_effect=ConnectionError("Redis unavailable"),
    )

    # Should not raise — errors are swallowed to avoid crashing the admin op
    await publish_registry_refresh_notification()


# ---------------------------------------------------------------------------
# subscribe_to_registry_refresh
# ---------------------------------------------------------------------------


def _make_pubsub(messages: list) -> MagicMock:
    """Build a mock pubsub that returns messages in sequence then raises CancelledError."""
    pubsub = AsyncMock()
    pubsub.subscribe = AsyncMock()
    # Once messages are exhausted the next get_message raises CancelledError to
    # break the infinite loop cleanly in tests.
    pubsub.get_message = AsyncMock(
        side_effect=messages + [asyncio.CancelledError()]
    )
    return pubsub


def _make_message(channel: str = REGISTRY_REFRESH_CHANNEL, msg_type: str = "message"):
    return {"type": msg_type, "channel": channel, "data": "refresh"}


@pytest.mark.asyncio
async def test_subscribe_calls_on_refresh_for_valid_message(mocker):
    """A message on the registry channel triggers the on_refresh callback."""
    on_refresh = AsyncMock()
    pubsub = _make_pubsub([_make_message()])

    mock_redis = MagicMock()
    mock_redis.pubsub.return_value = pubsub
    mocker.patch("redis.asyncio.Redis", return_value=mock_redis)

    await subscribe_to_registry_refresh(on_refresh)

    on_refresh.assert_called_once()


@pytest.mark.asyncio
async def test_subscribe_ignores_non_message_types(mocker):
    """Subscribe messages of type 'subscribe' (handshake) do not trigger on_refresh."""
    on_refresh = AsyncMock()
    pubsub = _make_pubsub([
        _make_message(msg_type="subscribe"),   # handshake — should be ignored
        _make_message(msg_type="psubscribe"),  # also ignored
    ])

    mock_redis = MagicMock()
    mock_redis.pubsub.return_value = pubsub
    mocker.patch("redis.asyncio.Redis", return_value=mock_redis)

    await subscribe_to_registry_refresh(on_refresh)

    on_refresh.assert_not_called()


@pytest.mark.asyncio
async def test_subscribe_ignores_wrong_channel(mocker):
    """Messages on a different channel do not trigger on_refresh."""
    on_refresh = AsyncMock()
    pubsub = _make_pubsub([_make_message(channel="some:other:channel")])

    mock_redis = MagicMock()
    mock_redis.pubsub.return_value = pubsub
    mocker.patch("redis.asyncio.Redis", return_value=mock_redis)

    await subscribe_to_registry_refresh(on_refresh)

    on_refresh.assert_not_called()


@pytest.mark.asyncio
async def test_subscribe_handles_none_message(mocker):
    """None returned by get_message (timeout) does not crash or trigger on_refresh."""
    on_refresh = AsyncMock()
    pubsub = _make_pubsub([None, None])  # two timeouts, then CancelledError

    mock_redis = MagicMock()
    mock_redis.pubsub.return_value = pubsub
    mocker.patch("redis.asyncio.Redis", return_value=mock_redis)

    await subscribe_to_registry_refresh(on_refresh)

    on_refresh.assert_not_called()


@pytest.mark.asyncio
async def test_subscribe_processes_multiple_messages(mocker):
    """Multiple valid messages each trigger on_refresh."""
    on_refresh = AsyncMock()
    pubsub = _make_pubsub([_make_message(), _make_message(), _make_message()])

    mock_redis = MagicMock()
    mock_redis.pubsub.return_value = pubsub
    mocker.patch("redis.asyncio.Redis", return_value=mock_redis)

    await subscribe_to_registry_refresh(on_refresh)

    assert on_refresh.call_count == 3


@pytest.mark.asyncio
async def test_subscribe_cancelled_error_stops_loop(mocker):
    """CancelledError at the outer loop level causes the function to return cleanly."""
    on_refresh = AsyncMock()
    pubsub = AsyncMock()
    pubsub.subscribe = AsyncMock(side_effect=asyncio.CancelledError())

    mock_redis = MagicMock()
    mock_redis.pubsub.return_value = pubsub
    mocker.patch("redis.asyncio.Redis", return_value=mock_redis)

    # Should return normally — CancelledError is caught and the loop breaks
    await subscribe_to_registry_refresh(on_refresh)

    on_refresh.assert_not_called()


@pytest.mark.asyncio
async def test_subscribe_reconnects_after_connection_error(mocker):
    """A connection error on the first attempt triggers a reconnect attempt."""
    on_refresh = AsyncMock()

    # First call raises ConnectionError; second call succeeds then CancelledError
    good_pubsub = _make_pubsub([_make_message()])
    bad_redis = MagicMock()
    bad_redis.pubsub.side_effect = ConnectionError("Redis down")
    good_redis = MagicMock()
    good_redis.pubsub.return_value = good_pubsub

    mock_redis_cls = mocker.patch(
        "redis.asyncio.Redis", side_effect=[bad_redis, good_redis]
    )
    mock_sleep = mocker.patch("asyncio.sleep", new=AsyncMock())

    await subscribe_to_registry_refresh(on_refresh)

    # Should have slept before retrying
    mock_sleep.assert_called_once_with(5)
    # Should have tried to create two Redis connections
    assert mock_redis_cls.call_count == 2
    # After reconnect, the valid message triggered on_refresh
    on_refresh.assert_called_once()
