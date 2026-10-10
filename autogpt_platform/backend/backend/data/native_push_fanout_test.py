import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from backend.api.model import NotificationPayload
from backend.data import notification_bus


@pytest.mark.asyncio
async def test_a_slow_web_push_provider_does_not_hold_up_native_push():
    release_web = asyncio.Event()
    native_delivered = asyncio.Event()

    async def slow_web(*args):
        await release_web.wait()

    async def native(*args):
        native_delivered.set()

    with (
        patch.object(
            notification_bus.AsyncRedisNotificationEventBus,
            "publish_event",
            AsyncMock(),
        ),
        patch.object(notification_bus, "send_push_for_user", slow_web),
        patch.object(notification_bus, "send_native_push_for_user", native),
    ):
        await notification_bus.AsyncRedisNotificationEventBus().publish(
            notification_bus.NotificationEvent(
                user_id="user",
                payload=NotificationPayload(type="attention", event="question"),
            )
        )
        try:
            await asyncio.wait_for(native_delivered.wait(), 0.2)
        finally:
            release_web.set()
            await asyncio.gather(*notification_bus._push_tasks)
