-- Consent captured at signup (SECRT-2783): which terms the account agreed to
-- and when, and a marketing opt-out that keeps it out of MailerLite.

-- Nullable with no default: a catalog-only change, no table rewrite.
-- AlterTable
ALTER TABLE "User" ADD COLUMN     "marketingOptOutAt" TIMESTAMP(3),
ADD COLUMN     "marketingOptOutSource" TEXT,
ADD COLUMN     "termsAcceptedAt" TIMESTAMP(3),
ADD COLUMN     "termsVersion" TEXT;
