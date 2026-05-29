-- Rename 'data' input to 'inputs' on all Agent Executor nodes
UPDATE "AgentNode" AS node
SET    "constantInput" = jsonb_set(
         "constantInput",
         '{inputs}',
         "constantInput"->'data'
       ) - 'data'
WHERE  node."agentBlockId" = 'e189baac-8c20-45a1-94a7-55177ea42565'
AND    node."constantInput" ? 'data';
