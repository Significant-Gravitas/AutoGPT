"""CRUD operations for Web Push subscriptions (PushSubscription model)."""

import logging
from datetime import datetime, timezone

from prisma.errors import RecordNotFoundError, UniqueViolationError
from prisma.models import PushSubscription
from pydantic import BaseModel

from backend.data.db import transaction
from backend.util.request import parse_url, validate_url_host

logger = logging.getLogger(__name__)


# Hostnames of legitimate Web Push services.  Endpoints submitted by
# clients must match one of these; everything else is rejected to prevent
# the backend (which POSTs to the stored URL via pywebpush) from being
# used as an SSRF primitive against internal infrastructure.  Chrome uses FCM
# (jmt17.google.com on pre-release channels), Firefox uses Autopush.
_PUSH_SERVICE_HOSTNAMES: list[str] = [
    "fcm.googleapis.com",
    "jmt17.google.com",
    "updates.push.services.mozilla.com",
]

# Edge on Windows (WNS) and Safari (Apple Web Push) issue endpoints on varying
# subdomains of a domain the vendor owns, so any subdomain of these is accepted.
_PUSH_SERVICE_DOMAIN_SUFFIXES = (".notify.windows.com", ".push.apple.com")

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
    hostname = parse_url(endpoint).hostname or ""
    vendor_subdomain = (
        [hostname] if hostname.endswith(_PUSH_SERVICE_DOMAIN_SUFFIXES) else []
    )
    parsed, is_trusted, _ = await validate_url_host(
        endpoint, trusted_hostnames=_PUSH_SERVICE_HOSTNAMES + vendor_subdomain
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
    try:
        return await _subscribe(user_id, endpoint, p256dh, auth, user_agent)
    except (UniqueViolationError, RecordNotFoundError):
        # Prisma runs the upsert as a read then a write, and a concurrent write
        # changed the endpoint's row in between: another subscribe created it,
        # or the failed-push cleanup deleted it. The transaction rolled back,
        # so one more run sees the row as it is now.
        return await _subscribe(user_id, endpoint, p256dh, auth, user_agent)


async def _subscribe(
    user_id: str,
    endpoint: str,
    p256dh: str,
    auth: str,
    user_agent: str | None,
) -> PushSubscription:
    """Check the cap, make room if needed and claim the endpoint, all or nothing.

    An endpoint belongs to one browser profile (``endpoint`` is unique), so
    whoever subscribed it last owns it. Signing into a second account on the
    same browser then stops the first account's notifications arriving there.
    """
    async with transaction() as tx:
        subscriptions = PushSubscription.prisma(tx)
        existing = await subscriptions.find_many(where={"userId": user_id})
        # The cap stops one account registering unbounded endpoints to amplify
        # outbound pushes. Refreshing its own endpoint is always fine. A new
        # endpoint past the cap is refused, but taking over one another
        # account registered (the same browser after switching accounts) is
        # not: refusing would leave that account's notifications arriving on
        # this browser. That makes room instead, dropping the device most
        # likely gone: the one failing most, else the least recently
        # refreshed. Failures bump updatedAt too, so it only measures a
        # refresh among rows that aren't failing (a subscribe resets
        # failCount).
        if len(existing) >= MAX_SUBSCRIPTIONS_PER_USER and not any(
            row.endpoint == endpoint for row in existing
        ):
            if await subscriptions.find_unique(where={"endpoint": endpoint}) is None:
                raise ValueError(
                    f"Subscription limit of {MAX_SUBSCRIPTIONS_PER_USER} "
                    "per user reached"
                )
            stalest = min(existing, key=lambda row: (-row.failCount, row.updatedAt))
            await subscriptions.delete_many(where={"id": stalest.id})
        return await subscriptions.upsert(
            where={"endpoint": endpoint},
            data={
                "create": {
                    "userId": user_id,
                    "endpoint": endpoint,
                    "p256dh": p256dh,
                    "auth": auth,
                    "userAgent": user_agent,
                },
                "update": {
                    "User": {"connect": {"id": user_id}},
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
