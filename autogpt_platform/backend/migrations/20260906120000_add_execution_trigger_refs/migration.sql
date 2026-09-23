-- AlterTable
-- Nullable columns with no default are catalog-only changes, so the
-- ACCESS EXCLUSIVE lock window on AgentGraphExecution is brief. IF NOT
-- EXISTS keeps a rerun from failing the deploy. No foreign keys: scheduler
-- jobs live in APScheduler's own store and webhooks are cleaned up
-- independently, so both are soft references like ActivityEvent.scheduleId.
ALTER TABLE "AgentGraphExecution" ADD COLUMN IF NOT EXISTS "scheduleId" TEXT;
ALTER TABLE "AgentGraphExecution" ADD COLUMN IF NOT EXISTS "webhookId" TEXT;
