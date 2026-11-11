-- Subscription invoices whose failed payment paused the plan, so paying them
-- later restarts the billing period from the payment (SECRT-2873). One row
-- per invoice records how far the restart got, so a repeated or delayed
-- Stripe event resumes it or finds it done.

CREATE TABLE "SubscriptionBillingRestart" (
    "stripeInvoiceId" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "stripeSubscriptionId" TEXT NOT NULL,
    "periodStart" TIMESTAMP(3) NOT NULL,
    "pausedAt" TIMESTAMP(3) NOT NULL,
    "paidAt" TIMESTAMP(3),
    "stripeCreditNoteId" TEXT,
    "restartedAt" TIMESTAMP(3),
    "abandonedAt" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "SubscriptionBillingRestart_pkey" PRIMARY KEY ("stripeInvoiceId")
);

CREATE INDEX "SubscriptionBillingRestart_userId_idx" ON "SubscriptionBillingRestart"("userId");

