-- Tier provenance: records WHY a user has their current subscriptionTier so
-- the bidirectional Stripe reconciliation can safely downgrade STRIPE-sourced
-- rows while leaving ADMIN/ENTERPRISE grants untouched.

-- CreateEnum
CREATE TYPE "SubscriptionTierSource" AS ENUM ('STRIPE', 'ADMIN', 'ENTERPRISE', 'SYSTEM');

-- AlterTable
ALTER TABLE "User"
  ADD COLUMN IF NOT EXISTS "subscriptionTierSource" "SubscriptionTierSource" NOT NULL DEFAULT 'SYSTEM',
  ADD COLUMN IF NOT EXISTS "lastStripeReconciledAt" TIMESTAMP(3);

-- Backfill provenance for existing rows. Idempotent: only rewrites rows still
-- at the SYSTEM default, so a re-run (or a row already classified) is a no-op.
-- Precedence: ENTERPRISE tier -> ENTERPRISE; otherwise a paid tier with a
-- Stripe customer -> STRIPE; a paid tier without a Stripe customer -> ADMIN;
-- everyone else (NO_TIER) stays SYSTEM.
UPDATE "User"
SET "subscriptionTierSource" = 'ENTERPRISE'
WHERE "subscriptionTier" = 'ENTERPRISE'
  AND "subscriptionTierSource" = 'SYSTEM';

UPDATE "User"
SET "subscriptionTierSource" = 'STRIPE'
WHERE "subscriptionTier" <> 'NO_TIER'
  AND "subscriptionTier" <> 'ENTERPRISE'
  AND "stripeCustomerId" IS NOT NULL
  AND "subscriptionTierSource" = 'SYSTEM';

UPDATE "User"
SET "subscriptionTierSource" = 'ADMIN'
WHERE "subscriptionTier" <> 'NO_TIER'
  AND "subscriptionTier" <> 'ENTERPRISE'
  AND "stripeCustomerId" IS NULL
  AND "subscriptionTierSource" = 'SYSTEM';
