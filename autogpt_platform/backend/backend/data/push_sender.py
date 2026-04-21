"""Fire-and-forget Web Push delivery for notification events."""

import asyncio
import json
import logging
import time
import uuid

from pywebpush import WebPushException, webpush

from backend.api.model import NotificationPayload
from backend.data.push_subscription import PushSubscriptionDTO
from backend.util.clients import get_database_manager_async_client
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

_settings = Settings()

# In-memory per-user debounce to collapse rapid-fire notifications
_user_last_push: dict[str, float] = {}
DEBOUNCE_SECONDS = 5.0

# Fields to forward from the notification payload to the push message
_FORWARDED_FIELDS = ("session_id", "step", "status", "graph_id", "execution_id")


def _build_push_payload(payload: NotificationPayload) -> str:
    """Build a compact JSON payload (<4KB) for the push message.

    ``id`` is a per-push UUID used by the service worker to build a unique
    notification tag, so repeat pushes don't get coalesced by the OS.
    """
    data = payload.model_dump(mode="json")
    compact: dict[str, object] = {
        "id": uuid.uuid4().hex,
        "type": data.get("type", ""),
        "event": data.get("event", ""),
    }
    for key in _FORWARDED_FIELDS:
        if key in data:
            compact[key] = data[key]
    return json.dumps(compact)


async def send_push_for_user(user_id: str, payload: NotificationPayload) -> None:
    """Send push notifications to all of a user's subscriptions.

    - Skips silently if VAPID keys are not configured.
    - Debounces per-user (collapses pushes within DEBOUNCE_SECONDS).
    - Cleans up stale subscriptions on 410/404 responses.
    """
    vapid_private = _settings.secrets.vapid_private_key
    vapid_public = _settings.secrets.vapid_public_key
    vapid_claim_email = _settings.secrets.vapid_claim_email
    if not vapid_private or not vapid_public or not vapid_claim_email:
        logger.debug("VAPID keys not fully configured, skipping push")
        return

    now = time.monotonic()
    last = _user_last_push.get(user_id, 0.0)
    if now - last < DEBOUNCE_SECONDS:
        logger.debug("Debouncing push for user %s", user_id)
        return
    _user_last_push[user_id] = now

    db_client = get_database_manager_async_client()
    subscriptions = await db_client.get_user_push_subscriptions(user_id)
    if not subscriptions:
        return

    push_data = _build_push_payload(payload)
    vapid_claims: dict[str, str | int] = {"sub": vapid_claim_email}

    async def _send_one(sub: PushSubscriptionDTO) -> None:
        try:
            await asyncio.to_thread(
                webpush,
                subscription_info={
                    "endpoint": sub.endpoint,
                    "keys": {"p256dh": sub.p256dh, "auth": sub.auth},
                },
                data=push_data,
                vapid_private_key=vapid_private,
                vapid_claims=vapid_claims,
            )
        except WebPushException as e:
            status = getattr(e.response, "status_code", None) if e.response else None
            if status in (410, 404):
                logger.info(
                    "Push subscription gone (%s), removing: %s",
                    status,
                    sub.endpoint[:60],
                )
                await db_client.delete_push_subscription_by_endpoint(
                    sub.user_id, sub.endpoint
                )
            else:
                logger.warning("Push failed for %s: %s", sub.endpoint[:60], e)
                await db_client.increment_push_fail_count(sub.user_id, sub.endpoint)
        except Exception:
            logger.exception("Unexpected error sending push to %s", sub.endpoint[:60])

    await asyncio.gather(
        *[_send_one(sub) for sub in subscriptions], return_exceptions=True
    )
