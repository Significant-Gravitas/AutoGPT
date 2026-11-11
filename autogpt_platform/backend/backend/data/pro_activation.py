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
    # The billing clock (backend/copilot/usage_clock.py), all Unix seconds in
    # UTC. Optional with defaults so a process on the previous release, on
    # either side of the database manager's RPC, still validates the state.
    created_at: int | None = None
    has_stripe_customer: bool = False
    # The Stripe billing period last copied from Stripe, if any.
    period_start: int | None = None
    period_end: int | None = None
    billing_anchor: int | None = None
    # The trial's own dates, only while the account is on the trial.
    trial_started_at: int | None = None
    trial_ends_at: int | None = None
    # Paid extra weeks that haven't expired or run out, taken together.
    extra_remaining: int = 0
    extra_total: int = 0
    extra_expires_at: int | None = None
    extra_count: int = 0


async def get_usage_activation_state(user_id: str) -> UsageActivationState:
    rows = await query_raw_with_schema(
        """SELECT u."id" AS user_id, a."id" AS generation,
        CASE WHEN u."subscriptionTier" = 'TRIAL' AND t."convertedAt" IS NULL AND t."consumedAt" IS NOT NULL
          AND t."status" = 'trialing' AND t."cardVerifiedAt" IS NOT NULL
          AND t."endsAt" > CURRENT_TIMESTAMP THEN t."id" END AS trial_id,
        u."subscriptionTier"::text AS tier,
        (a."id" IS NULL OR a."readyAt" IS NOT NULL) AS ready,
        FLOOR(EXTRACT(EPOCH FROM u."createdAt"))::bigint AS created_at,
        (u."stripeCustomerId" IS NOT NULL) AS has_stripe_customer,
        c."periodStart" AS period_start, c."periodEnd" AS period_end,
        c."billingCycleAnchor" AS billing_anchor,
        CASE WHEN u."subscriptionTier" = 'TRIAL'
          THEN FLOOR(EXTRACT(EPOCH FROM t."startedAt"))::bigint END AS trial_started_at,
        CASE WHEN u."subscriptionTier" = 'TRIAL'
          THEN FLOOR(EXTRACT(EPOCH FROM t."endsAt"))::bigint END AS trial_ends_at,
        COALESCE(x."remaining", 0)::bigint AS extra_remaining,
        COALESCE(x."total", 0)::bigint AS extra_total,
        x."expires_at" AS extra_expires_at,
        COALESCE(x."packs", 0)::int AS extra_count
        FROM {schema_prefix}"User" u
        LEFT JOIN {schema_prefix}"PaidUsageActivation" a ON a."userId" = u."id"
        LEFT JOIN {schema_prefix}"SubscriptionTrial" t ON t."userId" = u."id"
        LEFT JOIN {schema_prefix}"UsageBillingClock" c ON c."userId" = u."id"
        LEFT JOIN LATERAL (
          SELECT SUM(e."amountMicrodollars" - e."usedMicrodollars") AS "remaining",
            SUM(e."amountMicrodollars") AS "total",
            MIN(e."expiresAt") AS "expires_at", COUNT(*) AS "packs"
          FROM {schema_prefix}"UsageExtraAllowance" e
          WHERE e."userId" = u."id"
            AND e."usedMicrodollars" < e."amountMicrodollars"
            AND e."expiresAt" > EXTRACT(EPOCH FROM CURRENT_TIMESTAMP)::bigint
        ) x ON TRUE
        WHERE u."id" = $1""",
        user_id,
    )
    if len(rows) != 1:
        raise ValueError("Cannot establish usage entitlement for this user")
    return UsageActivationState.model_validate(rows[0])
