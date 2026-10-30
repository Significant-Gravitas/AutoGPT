-- Closing an account leaves 14 days to change your mind before its data is
-- erased (SECRT-2877). The table is new and small: the backfill only covers
-- rows written before this column existed.

-- AlterTable
ALTER TABLE "AccountDeletionRequest" ADD COLUMN "eraseAfter" TIMESTAMP(3);
UPDATE "AccountDeletionRequest" SET "eraseAfter" = "requestedAt" + INTERVAL '14 days';
ALTER TABLE "AccountDeletionRequest" ALTER COLUMN "eraseAfter" SET NOT NULL;
