-- Replace the 12-type notification surface with the four product-notification
-- families plus the account/billing service messages, and add the tables the
-- Alert and Briefing engines need.

-- 1. NotificationType: recreate rather than ALTER, because every old value is
--    gone. Pending rows in the old shape cannot be re-rendered by the new
--    templates, so the queue/batch tables are truncated as part of the swap.
TRUNCATE TABLE "NotificationEvent";
TRUNCATE TABLE "UserNotificationBatch" CASCADE;

ALTER TABLE "NotificationEvent" ALTER COLUMN "type" TYPE TEXT;
ALTER TABLE "UserNotificationBatch" ALTER COLUMN "type" TYPE TEXT;

DROP TYPE "NotificationType";

CREATE TYPE "NotificationType" AS ENUM (
  'BRIEFING',
  'ALERT',
  'VERDICT',
  'OPS',
  'SUBSCRIPTION_WELCOME',
  'PAYMENT_FAILED',
  'PAYMENT_FINAL_NOTICE',
  'SUBSCRIPTION_CANCELLED',
  'SUBSCRIPTION_RESUMED',
  'SUBSCRIPTION_ENDED'
);

ALTER TABLE "NotificationEvent"
  ALTER COLUMN "type" TYPE "NotificationType" USING "type"::"NotificationType";
ALTER TABLE "UserNotificationBatch"
  ALTER COLUMN "type" TYPE "NotificationType" USING "type"::"NotificationType";

-- 2. Preferences: the volume knob replaces the per-type checkbox list.
CREATE TYPE "BriefingFrequency" AS ENUM ('DAILY', 'WEEKLY', 'MONTHLY', 'OFF');

ALTER TABLE "User"
  ADD COLUMN "briefingFrequency" "BriefingFrequency" NOT NULL DEFAULT 'WEEKLY',
  ADD COLUMN "alertsEnabled" BOOLEAN NOT NULL DEFAULT true,
  ADD COLUMN "notifyOnStoreVerdict" BOOLEAN NOT NULL DEFAULT true,
  ADD COLUMN "lastBriefingAt" TIMESTAMP(3),
  ADD COLUMN "welcomeEmailSentAt" TIMESTAMP(3);

-- Carry over what the old columns can say about intent before dropping them:
-- someone who had turned the weekly summary off keeps their digest off, and
-- someone who had turned every alert-shaped notification off keeps alerts off.
UPDATE "User"
SET "briefingFrequency" = 'OFF'
WHERE "notifyOnWeeklySummary" = false;

UPDATE "User"
SET "alertsEnabled" = false
WHERE "notifyOnLowBalance" = false
  AND "notifyOnZeroBalance" = false;

UPDATE "User"
SET "notifyOnStoreVerdict" = false
WHERE "notifyOnAgentApproved" = false
  AND "notifyOnAgentRejected" = false;

ALTER TABLE "User"
  DROP COLUMN "notifyOnAgentRun",
  DROP COLUMN "notifyOnZeroBalance",
  DROP COLUMN "notifyOnLowBalance",
  DROP COLUMN "notifyOnBlockExecutionFailed",
  DROP COLUMN "notifyOnContinuousAgentError",
  DROP COLUMN "notifyOnDailySummary",
  DROP COLUMN "notifyOnWeeklySummary",
  DROP COLUMN "notifyOnMonthlySummary",
  DROP COLUMN "notifyOnAgentApproved",
  DROP COLUMN "notifyOnAgentRejected";

-- 3. Alert conditions.
CREATE TYPE "AlertCause" AS ENUM (
  'AUTH_EXPIRED',
  'PAUSED_FAILURES',
  'BLOCK_FAILED',
  'CONTINUOUS_ERROR',
  'AWAITING_REVIEW',
  'AWAITING_INPUT',
  'LOW_BALANCE',
  'ZERO_BALANCE',
  'GUARDRAIL'
);

CREATE TYPE "AlertConditionStatus" AS ENUM ('PENDING', 'SENT', 'DEFERRED', 'RESOLVED');

CREATE TABLE "AlertCondition" (
  "id"         TEXT NOT NULL,
  "createdAt"  TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updatedAt"  TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "userId"     TEXT NOT NULL,
  "cause"      "AlertCause" NOT NULL,
  "causeKey"   TEXT NOT NULL,
  "data"       JSONB NOT NULL,
  "status"     "AlertConditionStatus" NOT NULL DEFAULT 'PENDING',
  "sentAt"     TIMESTAMP(3),
  "resolvedAt" TIMESTAMP(3),
  "briefedAt"  TIMESTAMP(3),

  CONSTRAINT "AlertCondition_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX "AlertCondition_userId_causeKey_key"
  ON "AlertCondition"("userId", "causeKey");
CREATE INDEX "AlertCondition_status_createdAt_idx"
  ON "AlertCondition"("status", "createdAt");
CREATE INDEX "AlertCondition_userId_status_idx"
  ON "AlertCondition"("userId", "status");

ALTER TABLE "AlertCondition"
  ADD CONSTRAINT "AlertCondition_userId_fkey"
  FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- 4. Run interestingness, scored at completion so Briefing assembly is a
--    ranked read instead of a scan.
ALTER TABLE "AgentGraphExecution" ADD COLUMN "interestingness" DOUBLE PRECISION;

CREATE INDEX "AgentGraphExecution_userId_endedAt_interestingness_idx"
  ON "AgentGraphExecution"("userId", "endedAt", "interestingness");
