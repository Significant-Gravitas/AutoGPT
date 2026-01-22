-- Remove nodeId column and NodeExecution foreign key from PendingHumanReview
-- The auto-approval feature now uses ExecutionContext.auto_approved_node_ids instead of storing nodeId in the database.

-- Drop foreign key constraint (if exists) - this constraint linked PendingHumanReview to AgentNodeExecution via nodeExecId
ALTER TABLE "platform"."PendingHumanReview" DROP CONSTRAINT IF EXISTS "PendingHumanReview_nodeExecId_fkey";

-- Drop the nodeId column (if exists)
ALTER TABLE "platform"."PendingHumanReview" DROP COLUMN IF EXISTS "nodeId";
