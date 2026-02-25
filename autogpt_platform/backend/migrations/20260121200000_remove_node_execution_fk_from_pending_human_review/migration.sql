-- Remove NodeExecution foreign key from PendingHumanReview
-- The nodeExecId column remains as the primary key, but we remove the FK constraint
-- to AgentNodeExecution since PendingHumanReview records can persist after node
-- execution records are deleted.

-- Drop foreign key constraint that linked PendingHumanReview.nodeExecId to AgentNodeExecution.id
ALTER TABLE "PendingHumanReview" DROP CONSTRAINT IF EXISTS "PendingHumanReview_nodeExecId_fkey";
