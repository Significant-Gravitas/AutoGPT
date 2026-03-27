-- Rename STANDARD → PRO and add BUSINESS tier.
-- 1. Add BUSINESS to the enum
ALTER TYPE "SubscriptionTier" ADD VALUE IF NOT EXISTS 'BUSINESS';

-- 2. Migrate existing STANDARD users to PRO
UPDATE "User" SET "subscriptionTier" = 'PRO' WHERE "subscriptionTier" = 'STANDARD';

-- 3. Remove STANDARD from the enum by recreating the type.
-- PostgreSQL doesn't support DROP VALUE, so we recreate:
ALTER TYPE "SubscriptionTier" RENAME TO "SubscriptionTier_old";
CREATE TYPE "SubscriptionTier" AS ENUM ('FREE', 'PRO', 'BUSINESS', 'ENTERPRISE');
ALTER TABLE "User"
  ALTER COLUMN "subscriptionTier" TYPE "SubscriptionTier"
  USING "subscriptionTier"::text::"SubscriptionTier";
DROP TYPE "SubscriptionTier_old";

-- 4. Change default from FREE to PRO (beta testing: everyone gets PRO on sign-up)
ALTER TABLE "User" ALTER COLUMN "subscriptionTier" SET DEFAULT 'PRO';
