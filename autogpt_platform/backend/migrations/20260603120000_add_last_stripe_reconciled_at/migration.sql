-- Throttle timestamp for the lazy Stripe tier reconciliation staleness gate.
ALTER TABLE "User" ADD COLUMN IF NOT EXISTS "lastStripeReconciledAt" TIMESTAMP(3);
