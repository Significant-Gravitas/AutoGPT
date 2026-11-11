-- Usage limits on each customer's billing clock (SECRT-3441).
-- Three new tables, nothing altered: safe to apply with the previous release
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
CREATE TABLE "UsageExtraAllowance" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "purchaseKey" TEXT NOT NULL,
    "tier" "SubscriptionTier" NOT NULL,
    "amountMicrodollars" BIGINT NOT NULL,
    "usedMicrodollars" BIGINT NOT NULL DEFAULT 0,
    "priceCredits" INTEGER NOT NULL,
    "purchasedAt" BIGINT NOT NULL,
    "expiresAt" BIGINT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "UsageExtraAllowance_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "UsageExtraDebit" (
    "key" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "amountMicrodollars" BIGINT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "UsageExtraDebit_pkey" PRIMARY KEY ("key")
);

-- CreateIndex
CREATE INDEX "UsageExtraDebit_userId_idx" ON "UsageExtraDebit"("userId");

-- CreateIndex
CREATE UNIQUE INDEX "UsageExtraAllowance_userId_purchaseKey_key" ON "UsageExtraAllowance"("userId", "purchaseKey");

-- CreateIndex
CREATE INDEX "UsageExtraAllowance_userId_expiresAt_idx" ON "UsageExtraAllowance"("userId", "expiresAt");

-- AddForeignKey
ALTER TABLE "UsageBillingClock" ADD CONSTRAINT "UsageBillingClock_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "UsageExtraDebit" ADD CONSTRAINT "UsageExtraDebit_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "UsageExtraAllowance" ADD CONSTRAINT "UsageExtraAllowance_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
