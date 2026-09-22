-- Migrate existing users on BASIC (overloaded "no subscription" tier under
-- the old shape) to the new explicit NO_TIER. Users on PRO/MAX/BUSINESS/
-- ENTERPRISE keep their tier — they have active subs / admin grants.
-- Run after ADD VALUE 'NO_TIER' has committed (separate migration file
-- since Postgres requires the enum value to exist in a previously-
-- committed transaction before it can be referenced in DML).
UPDATE "User" SET "subscriptionTier" = 'NO_TIER' WHERE "subscriptionTier" = 'BASIC';
