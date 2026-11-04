-- AlterTable
ALTER TABLE "AgentGraphExecution"
    ADD COLUMN "triggerSource" TEXT,
    ADD COLUMN "triggerRef" TEXT;
