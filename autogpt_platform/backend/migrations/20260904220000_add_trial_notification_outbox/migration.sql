ALTER TABLE "SubscriptionTrial" ADD COLUMN "notificationRevision" INTEGER NOT NULL DEFAULT 0;
CREATE TABLE "TrialNotificationDelivery" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "trialId" TEXT NOT NULL REFERENCES "SubscriptionTrial"("id") ON DELETE CASCADE ON UPDATE CASCADE,
    "userId" TEXT NOT NULL,
    "idempotencyKey" TEXT NOT NULL,
    "payload" JSONB NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "attempts" INTEGER NOT NULL DEFAULT 0,
    "nextAttemptAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "nextWakeAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "leaseToken" TEXT,
    "leaseExpiresAt" TIMESTAMP(3),
    "providerMessageId" TEXT,
    "acceptedAt" TIMESTAMP(3),
    "lastError" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    CONSTRAINT "TrialNotificationDelivery_status_check" CHECK ("status" IN ('pending', 'sending', 'accepted', 'suppressed', 'failed')),
    CONSTRAINT "TrialNotificationDelivery_attempts_check" CHECK ("attempts" >= 0)
);
CREATE UNIQUE INDEX "TrialNotificationDelivery_idempotencyKey_key" ON "TrialNotificationDelivery"("idempotencyKey");
CREATE UNIQUE INDEX "TrialNotificationDelivery_providerMessageId_key" ON "TrialNotificationDelivery"("providerMessageId");
CREATE INDEX "TrialNotificationDelivery_status_nextWakeAt_idx" ON "TrialNotificationDelivery"("status", "nextWakeAt");
