-- AlterTable
ALTER TABLE "AgentGraphExecution" ADD COLUMN "expertId" TEXT;

-- AlterTable
ALTER TABLE "ExpertWorkflow" ADD COLUMN "scheduleCron" TEXT,
ADD COLUMN "scheduleId" TEXT;

-- CreateIndex
-- AgentGraphExecution is an existing, write-heavy table. Prisma's
-- `migrate deploy` wraps each migration in a single transaction, and
-- Postgres rejects `CREATE INDEX CONCURRENTLY` inside a transaction block;
-- the plain form briefly holds a ShareLock during the scan. For deployments
-- that can't tolerate that, run the CONCURRENTLY equivalent out-of-band
-- before this migration ships and Postgres will skip the recreate via
-- IF NOT EXISTS (same pattern as 20260727070306_add_expert_entity).
CREATE INDEX IF NOT EXISTS "AgentGraphExecution_expertId_createdAt_idx" ON "AgentGraphExecution"("expertId", "createdAt");

-- AddForeignKey
-- NOT VALID keeps the ACCESS EXCLUSIVE window on AgentGraphExecution to a
-- catalog update only — no full-table validation scan under lock. Validation
-- runs in the follow-up migration (separate transaction, SHARE UPDATE
-- EXCLUSIVE only), where it is trivially fast since every existing row has
-- a NULL expertId.
ALTER TABLE "AgentGraphExecution" ADD CONSTRAINT "AgentGraphExecution_expertId_fkey" FOREIGN KEY ("expertId") REFERENCES "Expert"("id") ON DELETE SET NULL ON UPDATE CASCADE NOT VALID;
