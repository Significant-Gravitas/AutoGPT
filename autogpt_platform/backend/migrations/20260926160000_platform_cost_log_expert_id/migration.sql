-- The expert a background LLM call ran for: a dream pass on an expert's
-- memory, a consult from an expert's chat. NULL for the account's own spend
-- and on every row written before this column.

-- Nullable with no default: a catalog-only change, no table rewrite.
ALTER TABLE "PlatformCostLog" ADD COLUMN "expertId" TEXT;

-- Plain CREATE INDEX: Prisma runs each migration in a transaction, which
-- rejects CONCURRENTLY. Where the brief write lock on PlatformCostLog matters,
-- build the same index CONCURRENTLY out of band first; IF NOT EXISTS then
-- skips it here.
CREATE INDEX IF NOT EXISTS "PlatformCostLog_expertId_createdAt_idx" ON "PlatformCostLog"("expertId", "createdAt");
