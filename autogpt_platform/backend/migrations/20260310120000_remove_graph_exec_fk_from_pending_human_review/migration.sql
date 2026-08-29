-- Remove GraphExecution foreign key from PendingHumanReview
-- The graphExecId column remains for querying, but we remove the FK constraint
-- to AgentGraphExecution since PendingHumanReview records can now be created
-- with synthetic graph_exec_ids (e.g., CoPilot direct block execution uses
-- "copilot-session-{session_id}" as graph_exec_id).

ALTER TABLE "PendingHumanReview" DROP CONSTRAINT IF EXISTS "PendingHumanReview_graphExecId_fkey";
