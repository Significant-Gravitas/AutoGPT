-- A deletion request made by email to privacy@ (SECRT-2854): the
-- `erase-account --close` command closes the account with this source.

-- Adding an enum value changes the catalog only: no table rewrite.
-- AlterEnum
ALTER TYPE "AccountDeletionSource" ADD VALUE 'EMAIL';
