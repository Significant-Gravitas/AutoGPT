-- Run after deploying code that accepts daily limits above weekly/total limits.
-- Preserve usage, dates and updatedAt (which also controls pending seat expiry).
UPDATE {schema_prefix}"SubscriptionTrial"
SET "offer" = "offer" || jsonb_build_object(
    'daily_cost_limit', 100000000,
    'weekly_cost_limit', 20000000,
    'total_cost_limit', 20000000
)
WHERE "convertedAt" IS NULL
  AND (
    ("status" = 'trialing' AND "endsAt" > CURRENT_TIMESTAMP)
    OR ("status" = 'checkout_pending' AND "consumedAt" IS NULL)
  )
  AND NOT "offer" @> jsonb_build_object(
    'daily_cost_limit', 100000000,
    'weekly_cost_limit', 20000000,
    'total_cost_limit', 20000000
  );
