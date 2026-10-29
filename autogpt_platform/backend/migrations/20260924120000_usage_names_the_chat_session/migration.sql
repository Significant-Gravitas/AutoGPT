-- AutoPilot chat usage was recorded under a synthetic graph execution
-- ("copilot-session-<session id>" in credit metadata, the bare session id in
-- the cost log); both now name the chat's session directly.

-- New rows only: the cost log holds a row per model call, so older chat rows
-- are read as chats by the admin view instead of being rewritten here. Nothing
-- filters on the column, so it has no index.
ALTER TABLE "PlatformCostLog" ADD COLUMN "chatSessionId" TEXT;

UPDATE "CreditTransaction"
SET metadata = (metadata - 'graph_exec_id' - 'graph_id'
                - CASE WHEN metadata->>'node_id' LIKE 'copilot-node-%' THEN 'node_id' ELSE '' END)
    || jsonb_build_object('chat_session_id', substr(metadata->>'graph_exec_id', length('copilot-session-') + 1))
WHERE metadata->>'graph_exec_id' LIKE 'copilot-session-%';

UPDATE "OrgCreditTransaction"
SET metadata = (metadata - 'graph_exec_id' - 'graph_id'
                - CASE WHEN metadata->>'node_id' LIKE 'copilot-node-%' THEN 'node_id' ELSE '' END)
    || jsonb_build_object('chat_session_id', substr(metadata->>'graph_exec_id', length('copilot-session-') + 1))
WHERE metadata->>'graph_exec_id' LIKE 'copilot-session-%';
