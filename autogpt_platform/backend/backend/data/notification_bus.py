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

# Allowlist of payload types that should trigger an OS-level web push.
# The notification bus is also used by in-page-only producers (e.g. onboarding
# step toasts) that we don't want surfacing as system notifications. Add new
# types here only when they're meant to reach users with the tab closed.
_PUSH_ENABLED_TYPES: frozenset[str] = frozenset({"copilot_completion"})


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
        # Fan out to web push only for payload types we've opted in. The bus
        # also carries in-page-only events (onboarding step toasts, etc.) that
        # shouldn't surface as OS notifications. Fire-and-forget so publishers
        # never wait on the push service; held in _push_tasks so the task
        # survives until completion.
        payload_type = event.payload.model_dump().get("type")
        if payload_type not in _PUSH_ENABLED_TYPES:
            return
        task = asyncio.create_task(_safe_send_push(event.user_id, event.payload))
        _push_tasks.add(task)
        task.add_done_callback(_push_tasks.discard)

    async def listen(self, user_id: str) -> AsyncGenerator[NotificationEvent, None]:
        """Stream notifications for a specific user."""
        async for event in self.listen_events(user_id):
            yield event
