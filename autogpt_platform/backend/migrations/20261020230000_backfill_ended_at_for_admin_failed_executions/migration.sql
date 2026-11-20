-- Admin cleanup paths marked executions FAILED without setting endedAt.
-- Backfill endedAt for terminal rows that are still missing it.
UPDATE "AgentGraphExecution"
SET "endedAt" = "updatedAt"
WHERE "endedAt" IS NULL
  AND "executionStatus" IN ('COMPLETED', 'FAILED', 'TERMINATED');
