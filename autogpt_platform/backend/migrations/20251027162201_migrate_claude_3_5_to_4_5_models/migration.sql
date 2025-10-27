-- Migrate Claude 3.5 models to Claude 4.5 models
-- This updates all AgentNode blocks that use deprecated Claude 3.5 models to the new 4.5 models
-- See: https://docs.anthropic.com/en/docs/about-claude/models/legacy-model-guide

-- Update Claude 3.5 Sonnet to Claude 4.5 Sonnet
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"claude-sonnet-4-5-20250929"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' = 'claude-3-5-sonnet-latest';

-- Update Claude 3.5 Haiku to Claude 4.5 Haiku
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"claude-haiku-4-5-20251001"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' = 'claude-3-5-haiku-latest';
