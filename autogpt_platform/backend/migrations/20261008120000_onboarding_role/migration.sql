-- The role picked in the onboarding wizard (SECRT-2852), kept apart from the
-- AutoPilot business understanding, which rewrites its own copy.

-- Nullable with no default: a catalog-only change, no table rewrite.
-- AlterTable
ALTER TABLE "UserOnboarding" ADD COLUMN     "role" TEXT,
ADD COLUMN     "roleOther" TEXT;
