-- Change the column default for new user rows from PRO to NO_TIER.
-- Existing rows are NOT modified — all historical PRO rows remain untouched.
ALTER TABLE "User" ALTER COLUMN "subscriptionTier" SET DEFAULT 'NO_TIER';
