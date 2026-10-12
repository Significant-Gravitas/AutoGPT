-- The expert a background LLM call ran for: a dream pass on an expert's
-- memory, a consult from an expert's chat. NULL for the account's own spend
-- and on every row written before this column.

-- Nullable with no default: a catalog-only change, no table rewrite, so the
-- ACCESS EXCLUSIVE lock ADD COLUMN takes is brief. The (expertId, createdAt)
-- index is built CONCURRENTLY by the next migration (20260926160001), never
-- under this lock. IF NOT EXISTS keeps a rerun, or a column an operator
-- already added out of band, from failing the deploy.
ALTER TABLE "PlatformCostLog" ADD COLUMN IF NOT EXISTS "expertId" TEXT;
