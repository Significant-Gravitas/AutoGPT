-- Migrate Claude 3.5 Sonnet blocks to Claude 4.5 Sonnet
-- This updates all AgentNode blocks that use deprecated Claude 3.5 models to the new 4.5 models
-- See: https://docs.anthropic.com/en/docs/about-claude/models/legacy-model-guide

-- Update Claude 3.5 Sonnet to Claude 4.5 Sonnet
WITH updated AS (
  UPDATE platform."AgentNode" AS node
  SET    "constantInput" = jsonb_set(
           "constantInput",
           '{model}',
           '"claude-sonnet-4-5-20250929"'
         )
  WHERE  node."constantInput"->>'model' = 'claude-3-5-sonnet-latest'
  RETURNING *
)
SELECT COUNT(*) FROM updated;

-- Update Claude 3.5 Haiku to Claude 4.5 Haiku
WITH updated AS (
  UPDATE platform."AgentNode" AS node
  SET    "constantInput" = jsonb_set(
           "constantInput",
           '{model}',
           '"claude-haiku-4-5-20251001"'
         )
  WHERE  node."constantInput"->>'model' = 'claude-3-5-haiku-latest'
  RETURNING *
)
SELECT COUNT(*) FROM updated;

COMMIT;
