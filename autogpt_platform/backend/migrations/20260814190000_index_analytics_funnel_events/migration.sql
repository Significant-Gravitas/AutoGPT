-- AnalyticsDetails is an existing write-heavy table. Prisma's `migrate deploy`
-- wraps each migration in a transaction, so Postgres rejects CONCURRENTLY here.
-- Deployments that cannot tolerate the inline locks should create this index
-- with CREATE INDEX CONCURRENTLY IF NOT EXISTS, then remove the old index with
-- DROP INDEX CONCURRENTLY IF EXISTS before applying this migration. These
-- idempotent statements then make the migration a no-op.
CREATE INDEX IF NOT EXISTS "AnalyticsDetails_userId_type_dataIndex_idx"
    ON "AnalyticsDetails"("userId", "type", "dataIndex");

DROP INDEX IF EXISTS "AnalyticsDetails_userId_type_idx";
