-- REL-002: durable cancellation — authoritative cancel request surviving executor restart
-- Adds cancelRequestedAt / cancelRequestedBy to AgentGraphExecution.
-- Backfill is not needed (existing rows null = not cancelled).
-- Rollback: DROP COLUMN if needed (no data loss beyond cancel state).
ALTER TABLE "AgentGraphExecution" ADD COLUMN IF NOT EXISTS "cancelRequestedAt" TIMESTAMP(3);
ALTER TABLE "AgentGraphExecution" ADD COLUMN IF NOT EXISTS "cancelRequestedBy" TEXT;
CREATE INDEX IF NOT EXISTS "AgentGraphExecution_cancelRequestedAt_idx" ON "AgentGraphExecution"("cancelRequestedAt");
