-- An operator can close and erase an account without the account-closed
-- email (`erase-account --close --now --no-email`, SECRT-3460). The request
-- records it, so the erasure does not send the email late.

-- A constant default: a catalog-only change, no table rewrite.
-- AlterTable
ALTER TABLE "AccountDeletionRequest" ADD COLUMN "confirmationSkipped" BOOLEAN NOT NULL DEFAULT false;
