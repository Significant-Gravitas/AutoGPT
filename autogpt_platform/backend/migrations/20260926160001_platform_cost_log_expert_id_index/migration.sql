-- CreateIndex
-- In its own migration so the build never runs under the ACCESS EXCLUSIVE
-- lock 20260926160000 takes to add the column. Run inline, the plain form
-- holds a SHARE lock on PlatformCostLog for the whole scan: reads continue,
-- cost-log writes wait. Postgres rejects CREATE INDEX CONCURRENTLY inside a
-- transaction block and Prisma wraps each migration in one, so deployments
-- that can't tolerate the write pause should build the CONCURRENTLY
-- equivalent out of band once 20260926160000 has landed the column; IF NOT
-- EXISTS then makes this statement a no-op.
CREATE INDEX IF NOT EXISTS "PlatformCostLog_expertId_createdAt_idx" ON "PlatformCostLog"("expertId", "createdAt");
