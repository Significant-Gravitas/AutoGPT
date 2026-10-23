"""Fire-and-forget Web Push delivery for notification events."""

import asyncio
import json
import logging
import re
import time
import uuid

from cachetools import TTLCache
from pywebpush import WebPushException, webpush

from backend.api.model import NotificationPayload
from backend.data.push_subscription import PushSubscriptionDTO, validate_push_endpoint
from backend.util.clients import get_database_manager_async_client
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

_settings = Settings()

DEBOUNCE_SECONDS = 5.0
# Per-user debounce timestamps, bounded + auto-evicted so the process doesn't
# accumulate one entry per user forever. Process-local — ineffective across
# multiple WS replicas; acceptable since debounce is a best-effort UX nicety.
_user_last_push = TTLCache[str, float](maxsize=10_000, ttl=DEBOUNCE_SECONDS)

# Fields to forward from the notification payload to the push message.
# Beyond the routing fields, the rich display fields (title/body/icon/url)
# and their structured context let the service worker render who did what
# without another fetch.
_FORWARDED_FIELDS = (
    "session_id",
    "step",
    "status",
    "graph_id",
    "execution_id",
    "title",
    "body",
    "icon",
    "url",
    "expert_name",
    "session_title",
)

# Byte budget for the encoded push payload. Web Push carries ~4 KB including
# content-coding overhead (~100 bytes for aes128gcm), and an oversized
# payload fails delivery — counting toward the failure janitor that deletes
# the subscription. Measured in UTF-8 bytes, not characters.
MAX_PUSH_BYTES = 3800

# Display text is capped for any publisher, mirroring the caps
# completion_notify composes with: unbounded input must never be what
# overflows the byte budget.
_TITLE_CHAR_CAP = 80
_BODY_CHAR_CAP = 160

# Optional fields, shed first (in order) when the payload would exceed the
# byte budget. The notification still renders from title/body, degrading to
# the default icon — and finally to the static click target — only if the
# earlier drops don't bring it under budget. Dropping `url` is safe: the
# service worker falls back to its static map (for copilot completions it
# rebuilds the identical session URL from `session_id`).
_DROPPABLE_FIELDS = ("session_title", "expert_name", "icon", "url")


def _extract_status_code(e: WebPushException) -> int | None:
    """Extract HTTP status code from a pywebpush exception."""
    if e.response is not None:
        return e.response.status_code
    # Fallback: parse "Push failed: <code> <reason>" out of the message in
    # case a future pywebpush version raises without attaching the Response.
    match = re.search(r"Push failed:\s*(\d{3})\b", str(e))
    return int(match.group(1)) if match else None


def _encode_push_payload(compact: dict[str, object]) -> str:
    # ensure_ascii=False: escaped \uXXXX sequences cost 6 bytes per CJK
    # character and 12 per emoji, while the UTF-8 bytes webpush sends cost 3
    # and 4. The service worker parses either form as JSON.
    return json.dumps(compact, ensure_ascii=False)


def _fit_push_payload(compact: dict[str, object]) -> str:
    """Serialize under MAX_PUSH_BYTES, capping display text and shedding context.

    Caps title/body first so no publisher's unbounded text can overflow the
    budget, then drops session_title, expert_name, icon and url until the
    UTF-8 encoded payload fits. The notification still renders from the
    capped title/body; an oversized payload would fail delivery and count
    toward the failure janitor that deletes the subscription.
    """
    for field, cap in (("title", _TITLE_CHAR_CAP), ("body", _BODY_CHAR_CAP)):
        text = compact.get(field)
        if isinstance(text, str) and len(text) > cap:
            compact[field] = text[: cap - 1].rstrip() + "…"
    for field in _DROPPABLE_FIELDS:
        payload_json = _encode_push_payload(compact)
        if len(payload_json.encode("utf-8")) <= MAX_PUSH_BYTES:
            return payload_json
        compact.pop(field, None)
    return _encode_push_payload(compact)


def _build_push_payload(payload: NotificationPayload) -> str:
    """Build a compact JSON payload for the push message.

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
    return _fit_push_payload(compact)


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
    # py_vapid rejects unprefixed strings deep in webpush(), where they'd
    # surface once per subscription as an "Unexpected error". Catch the
    # misconfiguration here and skip cleanly.
    if not vapid_claim_email.startswith(("mailto:", "https://")):
        logger.warning(
            "VAPID_CLAIM_EMAIL must start with 'mailto:' or 'https://', got %r — "
            "skipping push",
            vapid_claim_email[:40],
        )
        return

    if user_id in _user_last_push:
        logger.debug("Debouncing push for user %s", user_id)
        return
    _user_last_push[user_id] = time.monotonic()

    db_client = get_database_manager_async_client()
    subscriptions = await db_client.get_user_push_subscriptions(user_id)
    if not subscriptions:
        return

    push_data = _build_push_payload(payload)
    vapid_claims: dict[str, str | int] = {"sub": vapid_claim_email}

    async def _send_one(sub: PushSubscriptionDTO) -> None:
        try:
            # Defense-in-depth: reject endpoints that somehow bypassed the
            # subscribe-time check (rows written before the validator existed,
            # direct DB writes, or DNS changes that shifted a trusted host to
            # a blocked IP).
            await validate_push_endpoint(sub.endpoint)
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
        except ValueError as e:
            logger.warning(
                "Refusing push to untrusted endpoint %s: %s",
                sub.endpoint[:60],
                e,
            )
            await db_client.delete_push_subscription(sub.user_id, sub.endpoint)
            return
        except WebPushException as e:
            status = _extract_status_code(e)
            if status in (410, 404):
                logger.info(
                    "Push subscription gone (%s), removing: %s",
                    status,
                    sub.endpoint[:60],
                )
                await db_client.delete_push_subscription(sub.user_id, sub.endpoint)
            else:
                logger.warning("Push failed for %s: %s", sub.endpoint[:60], e)
                await db_client.increment_push_fail_count(sub.user_id, sub.endpoint)
        except Exception:
            logger.exception("Unexpected error sending push to %s", sub.endpoint[:60])

    await asyncio.gather(
        *[_send_one(sub) for sub in subscriptions], return_exceptions=True
    )
