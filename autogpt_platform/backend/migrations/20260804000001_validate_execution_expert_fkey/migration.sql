-- Validate the FK added NOT VALID in 20260804000000_add_execution_expert_attribution.
-- Runs in its own transaction with only a SHARE UPDATE EXCLUSIVE lock; every
-- pre-existing row has a NULL expertId so the scan is trivially fast.
ALTER TABLE "AgentGraphExecution" VALIDATE CONSTRAINT "AgentGraphExecution_expertId_fkey";
