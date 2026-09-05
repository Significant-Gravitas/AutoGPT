"""Durable trial-email intents and owner-fenced delivery leases."""

import logging
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from prisma.errors import UniqueViolationError
from prisma.models import SubscriptionTrial, TrialNotificationDelivery
from pydantic import BaseModel, Json

from backend.data.db import execute_raw_with_schema, query_raw_with_schema
from backend.data.notifications import TrialUpdateData
from backend.util.json import SafeJson

logger = logging.getLogger(__name__)
MAX_DELIVERY_ATTEMPTS = 8


class TrialDeliveryMessage(BaseModel):
    delivery_id: str


class TrialNotificationReceipt(BaseModel):
    id: str
    created: bool


class ClaimedTrialDelivery(BaseModel):
    id: str
    trial_id: str
    user_id: str
    payload: Json[TrialUpdateData] | TrialUpdateData
    attempts: int
    lease_token: str
    created_at: datetime


class DeliveryID(BaseModel):
    id: str


async def enqueue_trial_notification(
    user_id: str, trial_id: str, idempotency_key: str, data: TrialUpdateData
) -> TrialNotificationReceipt:
    trial = await SubscriptionTrial.prisma().find_unique_or_raise(
        where={"id": trial_id}
    )
    if trial.userId != user_id or not idempotency_key.startswith(f"trial:{trial_id}:"):
        raise ValueError("Trial notice ownership does not match enrollment")
    try:
        row = await TrialNotificationDelivery.prisma().create(
            data={
                "trialId": trial_id,
                "userId": user_id,
                "idempotencyKey": idempotency_key,
                "payload": SafeJson(data.model_dump(mode="json")),
                "nextAttemptAt": datetime.fromtimestamp(0, UTC),
                "nextWakeAt": datetime.fromtimestamp(0, UTC),
            }
        )
        return TrialNotificationReceipt(id=row.id, created=True)
    except UniqueViolationError:
        row = await TrialNotificationDelivery.prisma().find_unique_or_raise(
            where={"idempotencyKey": idempotency_key}
        )
    if row.trialId != trial_id or row.userId != user_id:
        raise ValueError("Trial notice key belongs to another enrollment")
    return TrialNotificationReceipt(id=row.id, created=False)


async def claim_trial_notification(delivery_id: str) -> ClaimedTrialDelivery | None:
    rows = await query_raw_with_schema(
        'UPDATE {schema_prefix}"TrialNotificationDelivery" SET '
        '"status" = \'sending\', "leaseToken" = $2, '
        "\"leaseExpiresAt\" = NOW() + INTERVAL '5 minutes', "
        '"attempts" = "attempts" + 1, "updatedAt" = NOW() '
        'WHERE "id" = $1 AND "attempts" < $3 AND ('
        '("status" = \'pending\' AND "nextAttemptAt" <= NOW()) OR '
        '("status" = \'sending\' AND "leaseExpiresAt" <= NOW())) '
        'RETURNING "id", "trialId" AS trial_id, "userId" AS user_id, '
        '"payload", "attempts", "leaseToken" AS lease_token, "createdAt" AS created_at',
        delivery_id,
        str(uuid4()),
        MAX_DELIVERY_ATTEMPTS,
        model=ClaimedTrialDelivery,
    )
    return rows[0] if rows else None


async def finish_trial_notification(
    delivery_id: str,
    lease_token: str,
    status: Literal["accepted", "suppressed"],
    provider_message_id: str | None = None,
) -> bool:
    if status == "accepted" and not provider_message_id:
        raise ValueError("Provider acceptance requires its message identity")
    count = await execute_raw_with_schema(
        'UPDATE {schema_prefix}"TrialNotificationDelivery" SET "status" = $3, '
        '"providerMessageId" = $4, "acceptedAt" = CASE WHEN $3 = \'accepted\' THEN NOW() ELSE NULL END, '
        '"leaseToken" = NULL, "leaseExpiresAt" = NULL, "lastError" = NULL, "updatedAt" = NOW() '
        'WHERE "id" = $1 AND "leaseToken" = $2 AND "status" = \'sending\'',
        delivery_id,
        lease_token,
        status,
        provider_message_id,
    )
    return count == 1


async def retry_trial_notification(
    delivery_id: str, lease_token: str, error_kind: str
) -> None:
    await execute_raw_with_schema(
        'UPDATE {schema_prefix}"TrialNotificationDelivery" SET '
        "\"status\" = CASE WHEN \"attempts\" >= $4 THEN 'failed' ELSE 'pending' END, "
        '"nextAttemptAt" = NOW() + LEAST(60 * POWER(2, LEAST("attempts", 6)), 3600) * INTERVAL \'1 second\', '
        '"nextWakeAt" = NOW() + LEAST(60 * POWER(2, LEAST("attempts", 6)), 3600) * INTERVAL \'1 second\', '
        '"leaseToken" = NULL, "leaseExpiresAt" = NULL, "lastError" = $3, "updatedAt" = NOW() '
        'WHERE "id" = $1 AND "leaseToken" = $2 AND "status" = \'sending\'',
        delivery_id,
        lease_token,
        error_kind[:100],
        MAX_DELIVERY_ATTEMPTS,
    )
    row = await TrialNotificationDelivery.prisma().find_unique(
        where={"id": delivery_id}
    )
    if row is not None and row.status == "failed":
        logger.error("Trial notice %s exhausted delivery attempts", delivery_id)


async def get_due_trial_notifications() -> list[str]:
    exhausted = await execute_raw_with_schema(
        'UPDATE {schema_prefix}"TrialNotificationDelivery" SET "status" = \'failed\', '
        '"lastError" = \'lease_expired_after_last_attempt\', "updatedAt" = NOW() '
        'WHERE "status" = \'sending\' AND "leaseExpiresAt" <= NOW() AND "attempts" >= $1',
        MAX_DELIVERY_ATTEMPTS,
    )
    if exhausted:
        logger.error("%s trial notices exhausted their final delivery lease", exhausted)
    rows = await query_raw_with_schema(
        'SELECT "id" FROM {schema_prefix}"TrialNotificationDelivery" '
        'WHERE "nextWakeAt" <= NOW() AND "attempts" < $1 AND ('
        '("status" = \'pending\' AND "nextAttemptAt" <= NOW()) OR '
        '("status" = \'sending\' AND "leaseExpiresAt" <= NOW())) '
        'ORDER BY "nextWakeAt", "id" LIMIT 100',
        MAX_DELIVERY_ATTEMPTS,
        model=DeliveryID,
    )
    return [row.id for row in rows]


async def mark_trial_notification_queued(delivery_id: str) -> None:
    await execute_raw_with_schema(
        'UPDATE {schema_prefix}"TrialNotificationDelivery" '
        'SET "nextWakeAt" = NOW() + INTERVAL \'5 minutes\', "updatedAt" = NOW() '
        "WHERE \"id\" = $1 AND \"status\" IN ('pending', 'sending')",
        delivery_id,
    )
