"""Tests for AsyncRedisNotificationEventBus.

Covers the tiny delegation surface: publish → publish_event(user_id), listen
→ listen_events(user_id), and the payload serializer that ensures extra
fields survive the Redis round-trip.
"""

from unittest.mock import AsyncMock, patch

import pytest

from backend.api.model import NotificationPayload
from backend.data.notification_bus import (
    AsyncRedisNotificationEventBus,
    NotificationEvent,
)


def test_notification_event_serializes_payload_including_extras():
    """``NotificationPayload`` allows extra fields; the bus serializer must
    preserve them. Dropping extras breaks feature payloads like CopilotCompletion."""
    payload = NotificationPayload(type="info", event="hey", extra_field="survive me")
    event = NotificationEvent(user_id="u", payload=payload)
    dumped = event.model_dump()
    assert dumped["payload"]["type"] == "info"
    assert dumped["payload"]["event"] == "hey"
    assert dumped["payload"]["extra_field"] == "survive me"


@pytest.mark.asyncio
async def test_publish_calls_publish_event_with_user_id_channel():
    """publish(event) → publish_event(event, channel_key=event.user_id)."""
    bus = AsyncRedisNotificationEventBus()
    event = NotificationEvent(
        user_id="user-42", payload=NotificationPayload(type="info", event="hi")
    )

    with patch.object(
        AsyncRedisNotificationEventBus, "publish_event", AsyncMock()
    ) as mock_pub:
        await bus.publish(event)

    mock_pub.assert_awaited_once()
    args = mock_pub.await_args.args
    # Pydantic may pass the event as a positional; regardless, user_id is the channel.
    assert args[-1] == "user-42"


@pytest.mark.asyncio
async def test_listen_delegates_to_listen_events_for_user():
    """listen(user_id) must subscribe on the per-user channel."""
    bus = AsyncRedisNotificationEventBus()

    captured: list[str] = []

    async def _listen_events(self, channel_key):
        captured.append(channel_key)
        if False:
            yield  # pragma: no cover

    with patch.object(AsyncRedisNotificationEventBus, "listen_events", _listen_events):
        async for _ in bus.listen("user-42"):
            break  # pragma: no cover — generator empty

    assert captured == ["user-42"]


def test_event_bus_name_is_configured() -> None:
    """The notification bus uses a distinct namespace from the execution bus,
    so WS exec channels and notification channels never collide."""
    bus = AsyncRedisNotificationEventBus()
    assert bus.event_bus_name  # non-empty, configured via Settings


@pytest.mark.asyncio
async def test_publish_fans_out_to_web_push():
    """publish() must also kick off web-push fanout for the user."""
    bus = AsyncRedisNotificationEventBus()
    event = NotificationEvent(
        user_id="user-42", payload=NotificationPayload(type="info", event="hi")
    )

    with (
        patch.object(AsyncRedisNotificationEventBus, "publish_event", AsyncMock()),
        patch(
            "backend.data.notification_bus.send_push_for_user",
            new_callable=AsyncMock,
        ) as mock_push,
    ):
        await bus.publish(event)
        # create_task is fire-and-forget — let the event loop drain the task.
        import asyncio

        for _ in range(3):
            await asyncio.sleep(0)

    mock_push.assert_awaited_once_with("user-42", event.payload)


@pytest.mark.asyncio
async def test_publish_skips_web_push_for_onboarding():
    """Onboarding step toasts are in-page only and must NOT trigger OS push."""
    bus = AsyncRedisNotificationEventBus()
    event = NotificationEvent(
        user_id="user-42",
        payload=NotificationPayload(type="onboarding", event="step_completed"),
    )

    with (
        patch.object(AsyncRedisNotificationEventBus, "publish_event", AsyncMock()),
        patch(
            "backend.data.notification_bus.send_push_for_user",
            new_callable=AsyncMock,
        ) as mock_push,
    ):
        await bus.publish(event)
        import asyncio

        for _ in range(3):
            await asyncio.sleep(0)

    mock_push.assert_not_awaited()


@pytest.mark.asyncio
async def test_publish_swallows_push_errors():
    """A failing push must not propagate or fail the publish."""
    bus = AsyncRedisNotificationEventBus()
    event = NotificationEvent(
        user_id="user-42", payload=NotificationPayload(type="info", event="hi")
    )

    with (
        patch.object(AsyncRedisNotificationEventBus, "publish_event", AsyncMock()),
        patch(
            "backend.data.notification_bus.send_push_for_user",
            new_callable=AsyncMock,
            side_effect=RuntimeError("push backend down"),
        ),
    ):
        await bus.publish(event)  # must not raise
        import asyncio

        for _ in range(3):
            await asyncio.sleep(0)
