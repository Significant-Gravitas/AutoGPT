-- Migrate deprecated OpenAI GPT-4-turbo and GPT-3.5-turbo models
-- This updates all AgentNode blocks that use deprecated models
-- OpenAI is retiring these models:
--   - gpt-4-turbo: March 26, 2026 -> migrate to gpt-4o
--   - gpt-3.5-turbo: September 28, 2026 -> migrate to gpt-4o-mini

-- Update gpt-4-turbo to gpt-4o (staying in same capability tier)
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"gpt-4o"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' = 'gpt-4-turbo';

-- Update gpt-3.5-turbo to gpt-4o-mini (appropriate replacement for lightweight model)
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"gpt-4o-mini"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' = 'gpt-3.5-turbo';

-- Update AgentPreset input overrides (stored in AgentNodeExecutionInputOutput)
UPDATE "AgentNodeExecutionInputOutput"
SET    "data" = JSONB_SET(
         "data"::jsonb,
         '{model}',
         '"gpt-4o"'::jsonb
       )
WHERE  "agentPresetId" IS NOT NULL
  AND  "data"::jsonb->>'model' = 'gpt-4-turbo';

UPDATE "AgentNodeExecutionInputOutput"
SET    "data" = JSONB_SET(
         "data"::jsonb,
         '{model}',
         '"gpt-4o-mini"'::jsonb
       )
WHERE  "agentPresetId" IS NOT NULL
  AND  "data"::jsonb->>'model' = 'gpt-3.5-turbo';
