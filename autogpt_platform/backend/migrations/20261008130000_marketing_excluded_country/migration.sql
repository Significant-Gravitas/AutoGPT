-- The first Iranian or Russian country a checkout saw for an account, which
-- keeps it out of MailerLite for good. Every later MailerLite write is checked
-- against it, since a trial or billing event no longer carries the IP country.

-- Nullable with no default: a catalog-only change, no table rewrite.
-- AlterTable
ALTER TABLE "User" ADD COLUMN     "marketingExcludedCountry" TEXT;
