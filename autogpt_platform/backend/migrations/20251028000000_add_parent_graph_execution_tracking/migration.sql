-- Add parent execution tracking for nested agent graphs
-- This enables cascading stop operations and prevents orphaned child executions

-- Add parentGraphExecutionId column to track parent-child relationships
ALTER TABLE "platform"."AgentGraphExecution"
ADD COLUMN "parentGraphExecutionId" TEXT;

-- Add foreign key constraint with SET NULL on delete
ALTER TABLE "platform"."AgentGraphExecution"
ADD CONSTRAINT "AgentGraphExecution_parentGraphExecutionId_fkey"
FOREIGN KEY ("parentGraphExecutionId")
REFERENCES "platform"."AgentGraphExecution"("id")
ON DELETE SET NULL
ON UPDATE CASCADE;

-- Add index for efficient child lookup queries
CREATE INDEX "AgentGraphExecution_parentGraphExecutionId_idx"
ON "platform"."AgentGraphExecution"("parentGraphExecutionId");
