"""Uncached, atomic billing/usage snapshot shared with execution workers."""

from prisma.enums import SubscriptionTier
from pydantic import BaseModel

from backend.data.db import query_raw_with_schema


class UsageActivationState(BaseModel):
    user_id: str
    generation: str | None = None
    trial_id: str | None = None
    tier: SubscriptionTier
    ready: bool


async def get_usage_activation_state(user_id: str) -> UsageActivationState:
    rows = await query_raw_with_schema(
        """SELECT u."id" AS user_id, a."id" AS generation,
        CASE WHEN u."subscriptionTier" = 'TRIAL' AND t."convertedAt" IS NULL AND t."consumedAt" IS NOT NULL
          AND t."status" = 'trialing' AND t."cardVerifiedAt" IS NOT NULL
          AND t."endsAt" > CURRENT_TIMESTAMP THEN t."id" END AS trial_id,
        u."subscriptionTier"::text AS tier,
        (a."id" IS NULL OR a."readyAt" IS NOT NULL) AS ready
        FROM {schema_prefix}"User" u
        LEFT JOIN {schema_prefix}"PaidUsageActivation" a ON a."userId" = u."id"
        LEFT JOIN {schema_prefix}"SubscriptionTrial" t ON t."userId" = u."id"
        WHERE u."id" = $1""",
        user_id,
    )
    if len(rows) != 1:
        raise ValueError("Cannot establish usage entitlement for this user")
    return UsageActivationState.model_validate(rows[0])
