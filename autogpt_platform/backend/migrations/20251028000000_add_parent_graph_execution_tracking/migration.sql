-- Add parent execution tracking for nested agent graphs
-- This enables cascading stop operations and prevents orphaned child executions

-- AlterTable
ALTER TABLE "AgentGraphExecution" ADD COLUMN "parentGraphExecutionId" TEXT;

-- CreateIndex
CREATE INDEX "AgentGraphExecution_parentGraphExecutionId_idx" ON "AgentGraphExecution"("parentGraphExecutionId");

-- AddForeignKey
ALTER TABLE "AgentGraphExecution" ADD CONSTRAINT "AgentGraphExecution_parentGraphExecutionId_fkey" FOREIGN KEY ("parentGraphExecutionId") REFERENCES "AgentGraphExecution"("id") ON DELETE SET NULL ON UPDATE CASCADE;
