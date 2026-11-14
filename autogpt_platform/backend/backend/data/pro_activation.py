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
    # The latest weekly window a paid usage reset was granted in, and how
    # many were granted in it. A reset applies only while that is the
    # account's current window.
    reset_week_start: int | None = None
    reset_count: int = 0


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
        r."weekStart" AS reset_week_start,
        COALESCE(r."resets", 0)::int AS reset_count
        FROM {schema_prefix}"User" u
        LEFT JOIN {schema_prefix}"PaidUsageActivation" a ON a."userId" = u."id"
        LEFT JOIN {schema_prefix}"SubscriptionTrial" t ON t."userId" = u."id"
        LEFT JOIN {schema_prefix}"UsageBillingClock" c ON c."userId" = u."id"
        LEFT JOIN LATERAL (
          SELECT g."weekStart", COUNT(*) AS "resets"
          FROM {schema_prefix}"UsageReset" g
          WHERE g."userId" = u."id" AND g."status" = 'granted'
          GROUP BY g."weekStart"
          ORDER BY g."weekStart" DESC
          LIMIT 1
        ) r ON TRUE
        WHERE u."id" = $1""",
        user_id,
    )
    if len(rows) != 1:
        raise ValueError("Cannot establish usage entitlement for this user")
    return UsageActivationState.model_validate(rows[0])
