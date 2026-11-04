"""Live task updates over the user notification bus.

The websocket server forwards every ``NotificationEvent`` on this bus to the
user's browser as ``{"method": "notification", "data": payload}``.
"""

import logging
from typing import Any

from backend.api.model import TaskEventNotificationPayload
from backend.data.notification_bus import (
    AsyncRedisNotificationEventBus,
    NotificationEvent,
)

logger = logging.getLogger(__name__)

_bus: AsyncRedisNotificationEventBus | None = None


def _get_bus() -> AsyncRedisNotificationEventBus:
    global _bus
    if _bus is None:
        _bus = AsyncRedisNotificationEventBus()
    return _bus


def _dump(value: Any) -> dict | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    return None


async def publish_task_event(
    user_id: str,
    event: str,
    task: Any | None,
    activity: Any | None = None,
    task_id: str | None = None,
) -> None:
    """Publish a ``task_event`` to the user's notification stream. Never raises."""
    try:
        task_json = _dump(task)
        activity_json = _dump(activity)
        resolved_id = (
            task_id
            or (task_json or {}).get("id")
            or (activity_json or {}).get("task_id")
        )
        if not resolved_id:
            logger.warning("Skipping task event %s without a task id", event)
            return
        payload = TaskEventNotificationPayload(
            type="task_event",
            event=event,
            task_id=resolved_id,
            task_key=(task_json or {}).get("key"),
            title=(task_json or {}).get("title"),
            task=task_json,
            activity=activity_json,
        )
        await _get_bus().publish(NotificationEvent(user_id=user_id, payload=payload))
    except Exception:
        logger.warning(
            "Failed to publish task event %s for task %s",
            event,
            task_id,
            exc_info=True,
        )
