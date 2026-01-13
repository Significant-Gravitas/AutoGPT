-- AlterTable
ALTER TABLE "AgentGraphExecution" ADD COLUMN "endedAt" TIMESTAMP(3);

-- Set endedAt to updatedAt for existing records
UPDATE "AgentGraphExecution" SET "endedAt" = "updatedAt" WHERE "endedAt" IS NULL;
