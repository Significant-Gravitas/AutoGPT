DROP TABLE IF EXISTS "TrialNotificationDelivery";

CREATE TABLE "SubscriptionTrialClaim" (
    "key" TEXT NOT NULL PRIMARY KEY,
    "trialId" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX "SubscriptionTrialClaim_trialId_idx" ON "SubscriptionTrialClaim"("trialId");
