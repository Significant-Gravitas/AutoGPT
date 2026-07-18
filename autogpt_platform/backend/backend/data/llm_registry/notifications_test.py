"""Tests for LLM registry pub/sub notifications (notifications.py).

Covers:
- publish_registry_refresh_notification: SPUBLISH happy path and Redis error
  swallowed
- subscribe_to_registry_refresh: smessage triggers on_refresh, handshake
  types ignored, CancelledError stops the loop, connection errors trigger
  reconnect, connections are closed on exit
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
async def test_publish_spublishes_on_registry_channel(mocker):
    """publish_registry_refresh_notification SPUBLISHes on the registry channel."""
    mock_cluster = AsyncMock()
    mocker.patch(
        "backend.data.llm_registry.notifications.get_redis_async",
        return_value=mock_cluster,
    )

    await publish_registry_refresh_notification()

    mock_cluster.execute_command.assert_called_once_with(
        "SPUBLISH", REGISTRY_REFRESH_CHANNEL, "refresh"
    )


@pytest.mark.asyncio
async def test_publish_swallows_redis_error(mocker):
    """Redis errors during publish are caught and logged, not raised."""
    mocker.patch(
        "backend.data.llm_registry.notifications.get_redis_async",
        side_effect=ConnectionError("Redis unavailable"),
    )

    # Should not raise — errors are swallowed to avoid crashing the admin op
    await publish_registry_refresh_notification()


# ---------------------------------------------------------------------------
# subscribe_to_registry_refresh
# ---------------------------------------------------------------------------


def _make_pubsub(messages: list) -> MagicMock:
    """Build a mock sharded pubsub that yields messages then cancels the loop."""
    pubsub = MagicMock()
    pubsub.execute_command = AsyncMock()  # SSUBSCRIBE
    pubsub.channels = {}
    pubsub.aclose = AsyncMock()

    async def listen():
        for m in messages:
            yield m
        raise asyncio.CancelledError()

    pubsub.listen = listen
    return pubsub


def _make_client(pubsub: MagicMock) -> MagicMock:
    client = MagicMock()
    client.pubsub.return_value = pubsub
    client.aclose = AsyncMock()
    return client


def _make_message(msg_type: str = "smessage"):
    return {"type": msg_type, "channel": REGISTRY_REFRESH_CHANNEL, "data": "refresh"}


@pytest.mark.asyncio
async def test_subscribe_calls_on_refresh_for_valid_message(mocker):
    """An smessage on the registry channel triggers the on_refresh callback."""
    on_refresh = AsyncMock()
    pubsub = _make_pubsub([_make_message()])
    client = _make_client(pubsub)
    mocker.patch(
        "backend.data.llm_registry.notifications.connect_sharded_pubsub_async",
        return_value=client,
    )

    await subscribe_to_registry_refresh(on_refresh)

    on_refresh.assert_called_once()
    pubsub.execute_command.assert_called_once_with(
        "SSUBSCRIBE", REGISTRY_REFRESH_CHANNEL
    )
    pubsub.aclose.assert_called_once()
    client.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_subscribe_ignores_handshake_types(mocker):
    """ssubscribe/subscribe handshake messages do not trigger on_refresh."""
    on_refresh = AsyncMock()
    pubsub = _make_pubsub(
        [
            _make_message(msg_type="ssubscribe"),
            _make_message(msg_type="subscribe"),
            None,
        ]
    )
    client = _make_client(pubsub)
    mocker.patch(
        "backend.data.llm_registry.notifications.connect_sharded_pubsub_async",
        return_value=client,
    )

    await subscribe_to_registry_refresh(on_refresh)

    on_refresh.assert_not_called()


@pytest.mark.asyncio
async def test_subscribe_processes_multiple_messages(mocker):
    """Multiple valid messages each trigger on_refresh."""
    on_refresh = AsyncMock()
    pubsub = _make_pubsub([_make_message(), _make_message(), _make_message()])
    client = _make_client(pubsub)
    mocker.patch(
        "backend.data.llm_registry.notifications.connect_sharded_pubsub_async",
        return_value=client,
    )

    await subscribe_to_registry_refresh(on_refresh)

    assert on_refresh.call_count == 3


@pytest.mark.asyncio
async def test_subscribe_on_refresh_error_does_not_kill_loop(mocker):
    """An exception inside on_refresh is logged; later messages still process."""
    on_refresh = AsyncMock(side_effect=[RuntimeError("boom"), None])
    pubsub = _make_pubsub([_make_message(), _make_message()])
    client = _make_client(pubsub)
    mocker.patch(
        "backend.data.llm_registry.notifications.connect_sharded_pubsub_async",
        return_value=client,
    )

    await subscribe_to_registry_refresh(on_refresh)

    assert on_refresh.call_count == 2


@pytest.mark.asyncio
async def test_subscribe_cancelled_during_subscribe_stops_loop(mocker):
    """CancelledError during SSUBSCRIBE returns cleanly and closes connections."""
    on_refresh = AsyncMock()
    pubsub = _make_pubsub([])
    pubsub.execute_command = AsyncMock(side_effect=asyncio.CancelledError())
    client = _make_client(pubsub)
    mocker.patch(
        "backend.data.llm_registry.notifications.connect_sharded_pubsub_async",
        return_value=client,
    )

    await subscribe_to_registry_refresh(on_refresh)

    on_refresh.assert_not_called()
    pubsub.aclose.assert_called_once()
    client.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_subscribe_reconnects_after_connection_error(mocker):
    """A connection error on the first attempt triggers a reconnect attempt."""
    on_refresh = AsyncMock()
    good_pubsub = _make_pubsub([_make_message()])
    good_client = _make_client(good_pubsub)

    mock_connect = mocker.patch(
        "backend.data.llm_registry.notifications.connect_sharded_pubsub_async",
        side_effect=[ConnectionError("Redis down"), good_client],
    )
    mock_sleep = mocker.patch(
        "backend.data.llm_registry.notifications.asyncio.sleep", new=AsyncMock()
    )

    await subscribe_to_registry_refresh(on_refresh)

    mock_sleep.assert_called_once_with(5)
    assert mock_connect.call_count == 2
    on_refresh.assert_called_once()
