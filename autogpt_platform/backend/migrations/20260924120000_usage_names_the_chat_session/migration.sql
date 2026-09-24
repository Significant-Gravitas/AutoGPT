-- AutoPilot chat usage was recorded under a synthetic graph execution
-- ("copilot-session-<session id>" in credit metadata, the bare session id in
-- the cost log); both now name the chat's session directly.
ALTER TABLE "PlatformCostLog" ADD COLUMN "sessionId" TEXT;

UPDATE "PlatformCostLog"
SET "sessionId" = "graphExecId", "graphExecId" = NULL
WHERE "blockId" = 'copilot' AND "graphExecId" IS NOT NULL;

CREATE INDEX "PlatformCostLog_sessionId_idx" ON "PlatformCostLog"("sessionId");

UPDATE "CreditTransaction"
SET metadata = (metadata - 'graph_exec_id' - 'graph_id'
                - CASE WHEN metadata->>'node_id' LIKE 'copilot-node-%' THEN 'node_id' ELSE '' END)
    || jsonb_build_object('session_id', substr(metadata->>'graph_exec_id', length('copilot-session-') + 1))
WHERE metadata->>'graph_exec_id' LIKE 'copilot-session-%';

UPDATE "OrgCreditTransaction"
SET metadata = (metadata - 'graph_exec_id' - 'graph_id'
                - CASE WHEN metadata->>'node_id' LIKE 'copilot-node-%' THEN 'node_id' ELSE '' END)
    || jsonb_build_object('session_id', substr(metadata->>'graph_exec_id', length('copilot-session-') + 1))
WHERE metadata->>'graph_exec_id' LIKE 'copilot-session-%';
