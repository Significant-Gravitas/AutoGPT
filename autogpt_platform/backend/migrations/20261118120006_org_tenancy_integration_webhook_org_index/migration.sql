-- CreateIndex
-- CONCURRENTLY and alone in its file, as 20261118120000 explains, which also
-- says how to recover from an interrupted build.
CREATE INDEX CONCURRENTLY IF NOT EXISTS "IntegrationWebhook_organizationId_idx" ON "IntegrationWebhook"("organizationId");
