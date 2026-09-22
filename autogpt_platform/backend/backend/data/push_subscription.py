"""CRUD operations for Web Push subscriptions (PushSubscription model)."""

import logging
from datetime import datetime, timezone

from prisma.models import PushSubscription
from pydantic import BaseModel

from backend.util.request import validate_url_host

logger = logging.getLogger(__name__)


# Hostnames of legitimate Web Push services.  Endpoints submitted by
# clients must match one of these; everything else is rejected to prevent
# the backend (which POSTs to the stored URL via pywebpush) from being
# used as an SSRF primitive against internal infrastructure.  Covers Chrome/
# Edge/Brave (FCM), Firefox (Autopush), and Safari/macOS (Apple Web Push).
_PUSH_SERVICE_HOSTNAMES: list[str] = [
    "fcm.googleapis.com",
    "updates.push.services.mozilla.com",
    "web.push.apple.com",
]

# Cap on concurrent push subscriptions per user — one entry per device/browser
# is typical, so this comfortably covers real usage while preventing an
# authenticated user from registering unbounded endpoints to amplify outbound
# traffic from the backend.
MAX_SUBSCRIPTIONS_PER_USER = 20

# Delete subscriptions with this many failed push attempts during periodic
# cleanup. Web Push sends occasionally fail transiently; beyond this threshold
# the endpoint is effectively dead and should be removed.
MAX_PUSH_FAILURES = 5


async def validate_push_endpoint(endpoint: str) -> None:
    """Ensure a push-subscription endpoint is an HTTPS URL hosted on a known
    Web Push provider.  Raises ``ValueError`` otherwise.

    Called at subscribe time and again before dispatch (defense-in-depth against
    rows written before this check existed or via future codepaths).
    """
    parsed, is_trusted, _ = await validate_url_host(
        endpoint, trusted_hostnames=_PUSH_SERVICE_HOSTNAMES
    )
    if parsed.scheme != "https":
        raise ValueError("Push endpoint must use https://")
    if not is_trusted:
        raise ValueError(
            f"Push endpoint host '{parsed.hostname}' is not a recognised "
            "Web Push service"
        )


class PushSubscriptionDTO(BaseModel):
    """RPC-serializable projection of PushSubscription."""

    user_id: str
    endpoint: str
    p256dh: str
    auth: str

    @staticmethod
    def from_db(model: PushSubscription) -> "PushSubscriptionDTO":
        return PushSubscriptionDTO(
            user_id=model.userId,
            endpoint=model.endpoint,
            p256dh=model.p256dh,
            auth=model.auth,
        )


async def upsert_push_subscription(
    user_id: str,
    endpoint: str,
    p256dh: str,
    auth: str,
    user_agent: str | None = None,
) -> PushSubscription:
    existing = await PushSubscription.prisma().find_many(
        where={"userId": user_id},
    )
    # Allow updates to an existing endpoint; only block when adding a *new* one
    # past the cap.
    has_this_endpoint = any(row.endpoint == endpoint for row in existing)
    if len(existing) >= MAX_SUBSCRIPTIONS_PER_USER and not has_this_endpoint:
        raise ValueError(
            f"Subscription limit of {MAX_SUBSCRIPTIONS_PER_USER} per user reached"
        )
    return await PushSubscription.prisma().upsert(
        where={"userId_endpoint": {"userId": user_id, "endpoint": endpoint}},
        data={
            "create": {
                "userId": user_id,
                "endpoint": endpoint,
                "p256dh": p256dh,
                "auth": auth,
                "userAgent": user_agent,
            },
            "update": {
                "p256dh": p256dh,
                "auth": auth,
                "userAgent": user_agent,
                "failCount": 0,
                "lastFailedAt": None,
            },
        },
    )


async def get_user_push_subscriptions(user_id: str) -> list[PushSubscriptionDTO]:
    rows = await PushSubscription.prisma().find_many(where={"userId": user_id})
    return [PushSubscriptionDTO.from_db(row) for row in rows]


async def delete_push_subscription(user_id: str, endpoint: str) -> None:
    await PushSubscription.prisma().delete_many(
        where={"userId": user_id, "endpoint": endpoint}
    )


async def increment_fail_count(user_id: str, endpoint: str) -> None:
    await PushSubscription.prisma().update_many(
        where={"userId": user_id, "endpoint": endpoint},
        data={
            "failCount": {"increment": 1},
            "lastFailedAt": datetime.now(timezone.utc),
        },
    )


async def cleanup_failed_subscriptions(
    max_failures: int = MAX_PUSH_FAILURES,
) -> int:
    """Delete subscriptions that have exceeded the failure threshold."""
    result = await PushSubscription.prisma().delete_many(
        where={"failCount": {"gte": max_failures}}
    )
    if result:
        logger.info(f"Cleaned up {result} failed push subscriptions")
    return result or 0
