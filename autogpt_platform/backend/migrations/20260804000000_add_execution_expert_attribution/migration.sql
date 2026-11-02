-- AlterTable
-- ADD COLUMN takes ACCESS EXCLUSIVE on AgentGraphExecution, but a nullable
-- column with no default is a catalog-only change, so the lock window is
-- brief. IF NOT EXISTS keeps a rerun — or an operator who already added the
-- column out-of-band — from failing the deploy with a duplicate-column
-- error. The (expertId, createdAt) index deliberately lives in a later
-- migration (20260804000003) so its full-table build never runs inside
-- this transaction's ACCESS EXCLUSIVE lock.
ALTER TABLE "AgentGraphExecution" ADD COLUMN IF NOT EXISTS "expertId" TEXT;

-- AlterTable
ALTER TABLE "ExpertWorkflow" ADD COLUMN IF NOT EXISTS "scheduleCron" TEXT;
ALTER TABLE "ExpertWorkflow" ADD COLUMN IF NOT EXISTS "scheduleId" TEXT;

-- AddForeignKey
-- NOT VALID keeps this a catalog update only — no full-table validation
-- scan under the lock. Validation runs in the follow-up migration (separate
-- transaction, SHARE UPDATE EXCLUSIVE only), where it is trivially fast
-- since every existing row has a NULL expertId.
ALTER TABLE "AgentGraphExecution" ADD CONSTRAINT "AgentGraphExecution_expertId_fkey" FOREIGN KEY ("expertId") REFERENCES "Expert"("id") ON DELETE SET NULL ON UPDATE CASCADE NOT VALID;
