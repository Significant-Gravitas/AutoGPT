"""CRUD operations for Web Push subscriptions (PushSubscription model)."""

import logging
from datetime import datetime, timezone

from prisma.models import PushSubscription
from pydantic import BaseModel

logger = logging.getLogger(__name__)


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


async def delete_push_subscription_by_endpoint(user_id: str, endpoint: str) -> None:
    """Remove a stale subscription (e.g. 410 Gone from push service)."""
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


async def cleanup_failed_subscriptions(max_failures: int = 5) -> int:
    """Delete subscriptions that have exceeded the failure threshold."""
    result = await PushSubscription.prisma().delete_many(
        where={"failCount": {"gte": max_failures}}
    )
    if result:
        logger.info(f"Cleaned up {result} failed push subscriptions")
    return result or 0
