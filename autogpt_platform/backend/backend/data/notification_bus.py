from __future__ import annotations

from typing import AsyncGenerator

from pydantic import BaseModel

from backend.data.event_bus import AsyncRedisEventBus
from backend.server.model import NotificationPayload
from backend.util.settings import Settings


class NotificationEvent(BaseModel):
    """Generic notification event destined for websocket delivery."""

    user_id: str
    payload: NotificationPayload


class AsyncRedisNotificationEventBus(AsyncRedisEventBus[NotificationEvent]):
    Model = NotificationEvent  # type: ignore

    @property
    def event_bus_name(self) -> str:
        return Settings().config.notification_event_bus_name

    async def publish(self, event: NotificationEvent) -> None:
        await self.publish_event(event, event.user_id)

    async def listen(
        self, user_id: str = "*"
    ) -> AsyncGenerator[NotificationEvent, None]:
        async for event in self.listen_events(user_id):
            yield event
