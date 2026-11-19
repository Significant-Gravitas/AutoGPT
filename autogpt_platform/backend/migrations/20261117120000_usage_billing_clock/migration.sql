-- Usage limits on each customer's billing clock (SECRT-3441).
-- Two new tables, nothing altered: safe to apply with the previous release
-- still running.

-- CreateTable
CREATE TABLE "UsageBillingClock" (
    "userId" TEXT NOT NULL,
    "stripeSubscriptionId" TEXT NOT NULL,
    "periodStart" BIGINT NOT NULL,
    "periodEnd" BIGINT NOT NULL,
    "billingCycleAnchor" BIGINT,
    "subscriptionCreatedAt" BIGINT NOT NULL,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "UsageBillingClock_pkey" PRIMARY KEY ("userId")
);

-- CreateTable
CREATE TABLE "UsageReset" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "purchaseKey" TEXT NOT NULL,
    "tier" "SubscriptionTier" NOT NULL,
    "priceCredits" INTEGER NOT NULL,
    "paidWith" TEXT NOT NULL,
    "status" TEXT NOT NULL,
    "stripeCheckoutSessionId" TEXT,
    "stripePaymentIntentId" TEXT,
    "weekStart" BIGINT,
    "weekEnd" BIGINT,
    "sequence" INTEGER,
    "grantedAt" BIGINT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "UsageReset_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "UsageReset_stripeCheckoutSessionId_key" ON "UsageReset"("stripeCheckoutSessionId");

-- CreateIndex
CREATE UNIQUE INDEX "UsageReset_stripePaymentIntentId_key" ON "UsageReset"("stripePaymentIntentId");

-- CreateIndex
CREATE UNIQUE INDEX "UsageReset_userId_purchaseKey_key" ON "UsageReset"("userId", "purchaseKey");

-- CreateIndex
CREATE UNIQUE INDEX "UsageReset_userId_weekStart_sequence_key" ON "UsageReset"("userId", "weekStart", "sequence");

-- CreateIndex
CREATE INDEX "UsageReset_userId_status_weekStart_idx" ON "UsageReset"("userId", "status", "weekStart");

-- AddForeignKey
ALTER TABLE "UsageBillingClock" ADD CONSTRAINT "UsageBillingClock_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "UsageReset" ADD CONSTRAINT "UsageReset_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
