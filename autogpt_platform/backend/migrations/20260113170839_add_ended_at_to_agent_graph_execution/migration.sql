-- AlterTable
ALTER TABLE "platform"."AgentGraphExecution" ADD COLUMN "endedAt" TIMESTAMP(3);

-- Set endedAt to updatedAt for existing records
UPDATE "platform"."AgentGraphExecution" SET "endedAt" = "updatedAt" WHERE "endedAt" IS NULL;
