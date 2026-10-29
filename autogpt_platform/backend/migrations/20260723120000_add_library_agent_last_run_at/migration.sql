-- Nullable "last run" timestamp for sorting the library by most recently run.
-- AlterTable
ALTER TABLE "LibraryAgent" ADD COLUMN "lastRunAt" TIMESTAMP(3);

-- Backfill from each agent's most recent execution so existing rows aren't all
-- NULL. Uses createdAt to match create_graph_execution's forward-path stamping.
UPDATE "LibraryAgent" la
SET "lastRunAt" = sub.last_run
FROM (
    SELECT "agentGraphId", "userId", MAX("createdAt") AS last_run
    FROM "AgentGraphExecution"
    GROUP BY "agentGraphId", "userId"
) sub
WHERE la."agentGraphId" = sub."agentGraphId"
  AND la."userId" = sub."userId";
