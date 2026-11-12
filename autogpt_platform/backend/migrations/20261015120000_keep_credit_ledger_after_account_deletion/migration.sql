-- Keep the credit ledger when an account is deleted (SECRT-2854). It is a
-- billing record, kept for six years, and #10917's ON DELETE NO ACTION meant
-- to keep it but instead refused to delete an account that had any row. The
-- rows keep their user ID as a plain column; account deletion strips the
-- personal content from their metadata.

-- Dropping a constraint changes the catalog only: no table rewrite or scan.
-- DropForeignKey
ALTER TABLE "CreditTransaction" DROP CONSTRAINT "CreditTransaction_userId_fkey";
