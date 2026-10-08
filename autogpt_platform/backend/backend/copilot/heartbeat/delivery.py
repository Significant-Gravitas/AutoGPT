"""Where a heartbeat alert goes: the main thread, the notification, the bots.

Each target is best effort and independent: one that fails is logged and the
others still go, so a bot outage cannot swallow the alert in the web app.
"""

import logging
import uuid

from pydantic import BaseModel

from backend.api.model import CopilotCompletionPayload
from backend.data.db_accessors import chat_db
from backend.data.notification_bus import (
    AsyncRedisNotificationEventBus,
    NotificationEvent,
)

from .config import HeartbeatDelivery

logger = logging.getLogger(__name__)

# Fixed so a retried delivery of one run derives the same message id, which
# ``append_plain_session_message`` dedupes on.
_NAMESPACE = uuid.UUID("5b0c6f0e-4f43-4f8e-9a52-0c1f7f6a3d21")
ALERT_MESSAGE_KIND = "heartbeat_alert"


class DeliveryReport(BaseModel):
    thread_session_id: str | None = None
    notified: bool = False
    chat_platforms: list[str] = []


async def deliver_alert(
    user_id: str,
    run_session_id: str,
    text: str,
    targets: HeartbeatDelivery,
) -> DeliveryReport:
    report = DeliveryReport()
    if targets.copilot_thread:
        report.thread_session_id = await _post_to_thread(user_id, run_session_id, text)
    # Without a thread post the notification opens the heartbeat's own chat,
    # which holds the turn that raised the alert.
    report.notified = await _notify(
        user_id,
        report.thread_session_id or run_session_id,
        text,
        push=targets.push,
    )
    for platform in targets.chat_platforms:
        if await _post_to_chat_platform(user_id, platform, text):
            report.chat_platforms.append(platform)
    return report


async def _post_to_thread(user_id: str, run_session_id: str, text: str) -> str | None:
    message_id = str(uuid.uuid5(_NAMESPACE, f"heartbeat:{user_id}:{run_session_id}"))
    try:
        return await chat_db().append_plain_session_message(
            user_id=user_id,
            content=text,
            message_id=message_id,
            metadata={
                "kind": ALERT_MESSAGE_KIND,
                "heartbeat_session_id": run_session_id,
            },
        )
    except Exception:
        logger.warning(
            "Heartbeat alert for user %s could not be posted to the thread",
            user_id[:12],
            exc_info=True,
        )
        return None


async def _notify(user_id: str, session_id: str, text: str, *, push: bool) -> bool:
    """The copilot notification the web app already handles; the bus also
    sends it as a web push unless the user turned push off."""
    event = NotificationEvent(
        user_id=user_id,
        # Validated from a dict: ``source`` and ``preview`` ride the
        # payload's extra fields, which the web app may read and the
        # existing handlers ignore.
        payload=CopilotCompletionPayload.model_validate(
            {
                "type": "copilot_completion",
                "event": "session_completed",
                "session_id": session_id,
                "status": "completed",
                "source": "heartbeat",
                "preview": text[:200],
            }
        ),
    )
    bus = AsyncRedisNotificationEventBus()
    try:
        if push:
            await bus.publish(event)
        else:
            await bus.publish_event(event, user_id)
        return True
    except Exception:
        logger.warning(
            "Heartbeat notification for user %s failed", user_id[:12], exc_info=True
        )
        return False


async def _post_to_chat_platform(user_id: str, platform_name: str, text: str) -> bool:
    """DM the user on a linked chat platform, as ``post_to_chat_platform``
    with ``target='dm'`` does; the bridge refuses a user with no DM link."""
    # Deferred: the tools package imports the whole registry.
    from backend.copilot.tools.chat_platform import (
        _any_chat_platform_configured,
        _resolve_platform,
    )
    from backend.util.clients import get_copilot_chat_bridge_client

    if not _any_chat_platform_configured():
        return False
    platform, _ = _resolve_platform(platform_name)
    if platform is None:
        return False
    try:
        result = await get_copilot_chat_bridge_client().send_dm_to_user(
            platform=platform, user_id=user_id, content=text
        )
    except Exception:
        logger.warning(
            "Heartbeat alert for user %s could not reach %s",
            user_id[:12],
            platform_name,
            exc_info=True,
        )
        return False
    if not result.ok:
        logger.info(
            "Heartbeat alert for user %s not delivered to %s: %s",
            user_id[:12],
            platform_name,
            result.error,
        )
    return bool(result.ok)
