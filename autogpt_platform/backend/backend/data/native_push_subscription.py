from datetime import datetime, timezone
from typing import Literal

from prisma.models import NativePushSubscription
from pydantic import BaseModel


class NativePushSubscriptionDTO(BaseModel):
    id: str
    provider: Literal["apns", "fcm"]
    token: str
    environment: Literal["sandbox", "production"]
    origin: str


async def get_native_push_subscriptions(
    user_id: str,
) -> list[NativePushSubscriptionDTO]:
    now = datetime.now(timezone.utc)
    rows = await NativePushSubscription.prisma().find_many(
        where={
            "session": {
                "is": {
                    "userId": user_id,
                    "expiresAt": {"gt": now},
                    "impersonatedBy": None,
                    "user": {
                        "is": {
                            "OR": [
                                {"banned": False},
                                {"banned": None},
                                {"banExpires": {"lte": now}},
                            ]
                        }
                    },
                }
            }
        },
        take=20,
        order={"updatedAt": "desc"},
    )
    return [
        NativePushSubscriptionDTO.model_validate(
            {
                "id": row.id,
                "provider": row.provider,
                "token": row.token,
                "environment": row.environment,
                "origin": row.origin,
            }
        )
        for row in rows
    ]


async def delete_native_push_subscription(binding_id: str) -> None:
    await NativePushSubscription.prisma().delete_many(where={"id": binding_id})
