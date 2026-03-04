-- AlterTable
ALTER TABLE "AgentGraphExecution" ADD COLUMN "endedAt" TIMESTAMP(3);

-- Set endedAt to updatedAt for existing records with terminal status only
UPDATE "AgentGraphExecution"
SET "endedAt" = "updatedAt"
WHERE "endedAt" IS NULL
  AND "executionStatus" IN ('COMPLETED', 'FAILED', 'TERMINATED');
