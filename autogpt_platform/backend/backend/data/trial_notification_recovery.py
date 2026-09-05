"""Find trial state whose customer notices were never recorded."""

from pydantic import BaseModel

from backend.data.db import query_raw_with_schema


class TrialNoticeCandidate(BaseModel):
    id: str
    user_id: str
    subscription_id: str


async def get_trial_notice_candidates(after_id: str = "") -> list[TrialNoticeCandidate]:
    return await query_raw_with_schema(
        """
        SELECT t."id", t."userId" AS user_id,
               t."stripeSubscriptionId" AS subscription_id
        FROM {schema_prefix}"SubscriptionTrial" t
        WHERE t."id" > $1 AND t."consumedAt" IS NOT NULL
          AND t."stripeSubscriptionId" IS NOT NULL AND t."endsAt" IS NOT NULL
          AND (
            (t."status" = 'trialing' AND t."endsAt" <= NOW()
             AND t."convertedAt" IS NULL)
            OR EXISTS (
              SELECT 1 FROM (VALUES
                ('started', t."status" = 'trialing' AND t."endsAt" > NOW()
                  AND NOT t."cancelAtPeriodEnd" AND t."cardVerifiedAt" IS NOT NULL
                  AND t."convertedAt" IS NULL),
                ('ending:' || FLOOR(EXTRACT(EPOCH FROM t."endsAt"))::bigint::text,
                  t."status" = 'trialing' AND t."endsAt" > NOW()
                  AND t."endsAt" <= NOW() + INTERVAL '3 days'
                  AND NOT t."cancelAtPeriodEnd" AND t."cardVerifiedAt" IS NOT NULL
                  AND t."convertedAt" IS NULL),
                ('canceled:' || t."notificationRevision"::text,
                  t."status" = 'trialing' AND t."endsAt" > NOW()
                  AND t."cancelAtPeriodEnd" AND t."convertedAt" IS NULL),
                ('resumed:' || t."notificationRevision"::text,
                  t."status" = 'trialing' AND t."endsAt" > NOW()
                  AND NOT t."cancelAtPeriodEnd" AND t."notificationRevision" > 0
                  AND t."cardVerifiedAt" IS NOT NULL AND t."convertedAt" IS NULL),
                ('payment_failed', t."status" IN ('past_due', 'unpaid', 'incomplete')
                  AND t."convertedAt" IS NULL),
                ('ended', t."status" IN ('canceled', 'paused')
                  AND t."convertedAt" IS NULL),
                ('converted:' || t."stripeConversionInvoiceId",
                  t."status" = 'active' AND t."convertedAt" IS NOT NULL
                  AND t."stripeConversionInvoiceId" IS NOT NULL)
              ) AS expected(suffix, applies)
              WHERE expected.applies AND NOT EXISTS (
                SELECT 1 FROM {schema_prefix}"TrialNotificationDelivery" d
                WHERE d."idempotencyKey" = 'trial:' || t."id" || ':' || expected.suffix
                  AND (d."status" <> 'suppressed'
                    OR d."providerMessageId" IS NOT NULL OR d."acceptedAt" IS NOT NULL
                    OR d."leaseToken" IS NOT NULL OR d."leaseExpiresAt" IS NOT NULL)
              )
            )
          )
        ORDER BY t."id" LIMIT 100
        """,
        after_id,
        model=TrialNoticeCandidate,
    )
