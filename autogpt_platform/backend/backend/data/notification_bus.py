from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

from pydantic import BaseModel, field_serializer

from backend.api.model import NotificationPayload
from backend.data.event_bus import AsyncRedisEventBus
from backend.data.push_sender import send_push_for_user
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
_settings = Settings()

# Strong refs for in-flight push fanout tasks. asyncio only keeps weak refs
# to tasks, so a fire-and-forget create_task can be GC'd mid-run.
_push_tasks: set[asyncio.Task] = set()


class NotificationEvent(BaseModel):
    """Generic notification event destined for websocket delivery."""

    user_id: str
    payload: NotificationPayload

    @field_serializer("payload")
    def serialize_payload(self, payload: NotificationPayload):
        """Ensure extra fields survive Redis serialization."""
        return payload.model_dump()


async def _safe_send_push(user_id: str, payload: NotificationPayload) -> None:
    """Deliver web push for a notification, swallowing errors."""
    try:
        await send_push_for_user(user_id, payload)
    except Exception:
        logger.exception("Failed to send web push for user %s", user_id)


class AsyncRedisNotificationEventBus(AsyncRedisEventBus[NotificationEvent]):
    Model = NotificationEvent  # type: ignore

    @property
    def event_bus_name(self) -> str:
        return _settings.config.notification_event_bus_name

    async def publish(self, event: NotificationEvent) -> None:
        await self.publish_event(event, event.user_id)
        # Skip OS push for onboarding step toasts — those are in-page only.
        # TODO: remove once the onboarding/wallet rework lands and decides
        # per-event whether a system notification is desired.
        if event.payload.model_dump().get("type") == "onboarding":
            return
        # Fan out to web push subscriptions in parallel. Fire-and-forget so
        # publishers never wait on the push service; held in _push_tasks so
        # the task survives until completion.
        task = asyncio.create_task(_safe_send_push(event.user_id, event.payload))
        _push_tasks.add(task)
        task.add_done_callback(_push_tasks.discard)

    async def listen(self, user_id: str) -> AsyncGenerator[NotificationEvent, None]:
        """Stream notifications for a specific user."""
        async for event in self.listen_events(user_id):
            yield event
