-- Add SubscriptionTier.NO_TIER as the explicit "user has no active Stripe
-- subscription" state. Previously BASIC was overloaded for both this
-- semantic AND a potential entry-paid-tier slot; NO_TIER disambiguates.
-- BASIC stays in the enum as a future paid-tier option.
--
-- ADD VALUE BEFORE 'BASIC' keeps it as the lowest tier for ordering. RENAME
-- existing BASIC users to NO_TIER in the same migration so everyone who
-- previously had no-subscription state lands in the new explicit state.
ALTER TYPE "SubscriptionTier" ADD VALUE IF NOT EXISTS 'NO_TIER' BEFORE 'BASIC';
