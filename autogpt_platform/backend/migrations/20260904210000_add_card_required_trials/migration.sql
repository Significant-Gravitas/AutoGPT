ALTER TYPE "SubscriptionTier" ADD VALUE 'TRIAL';
ALTER TYPE "NotificationType" ADD VALUE 'TRIAL_UPDATE';

CREATE TABLE "SubscriptionTrial" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "offer" JSONB NOT NULL,
    "stripeCustomerId" TEXT NOT NULL,
    "stripeCheckoutSessionId" TEXT,
    "stripeSubscriptionId" TEXT,
    "checkoutAttempt" INTEGER NOT NULL DEFAULT 0,
    "checkoutSuccessUrl" TEXT NOT NULL,
    "checkoutCancelUrl" TEXT NOT NULL,
    "checkoutMetadata" JSONB NOT NULL DEFAULT '{}',
    "status" TEXT NOT NULL DEFAULT 'checkout_pending',
    "cardVerifiedAt" TIMESTAMP(3),
    "startedAt" TIMESTAMP(3),
    "endsAt" TIMESTAMP(3),
    "consumedAt" TIMESTAMP(3),
    "convertedAt" TIMESTAMP(3),
    "cancelAtPeriodEnd" BOOLEAN NOT NULL DEFAULT false,
    "costMicrodollars" BIGINT NOT NULL DEFAULT 0,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    CONSTRAINT "SubscriptionTrial_pkey" PRIMARY KEY ("id"),
    CONSTRAINT "SubscriptionTrial_cost_nonnegative" CHECK ("costMicrodollars" >= 0)
);

CREATE UNIQUE INDEX "SubscriptionTrial_userId_key" ON "SubscriptionTrial"("userId");
CREATE UNIQUE INDEX "SubscriptionTrial_stripeCheckoutSessionId_key" ON "SubscriptionTrial"("stripeCheckoutSessionId");
CREATE UNIQUE INDEX "SubscriptionTrial_stripeSubscriptionId_key" ON "SubscriptionTrial"("stripeSubscriptionId");
CREATE INDEX "SubscriptionTrial_status_endsAt_idx" ON "SubscriptionTrial"("status", "endsAt");
ALTER TABLE "SubscriptionTrial" ADD CONSTRAINT "SubscriptionTrial_userId_fkey"
    FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
