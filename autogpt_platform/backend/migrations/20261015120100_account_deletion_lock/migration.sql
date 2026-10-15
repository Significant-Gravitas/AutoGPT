-- When `delete-account --lock` stopped an account (SECRT-2854). The full
-- deletion waits out a 14-day undo window from then; `--unlock` clears it.

-- Nullable with no default: a catalog-only change, no table rewrite.
-- AlterTable
ALTER TABLE "User" ADD COLUMN     "deletionLockedAt" TIMESTAMP(3);
