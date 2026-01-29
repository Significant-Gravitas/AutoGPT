-- Migrate Claude 3.7 Sonnet to Claude 4.5 Sonnet
-- This updates all AgentNode blocks that use the deprecated Claude 3.7 Sonnet model
-- Anthropic is retiring claude-3-7-sonnet-20250219 on February 19, 2026

-- Update AgentNode constant inputs
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"claude-sonnet-4-5-20250929"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' = 'claude-3-7-sonnet-20250219';

-- Update AgentPreset input overrides (stored in AgentNodeExecutionInputOutput)
UPDATE "AgentNodeExecutionInputOutput"
SET    "data" = JSONB_SET(
         "data"::jsonb,
         '{model}',
         '"claude-sonnet-4-5-20250929"'::jsonb
       )
WHERE  "agentPresetId" IS NOT NULL
  AND  "data"::jsonb->>'model' = 'claude-3-7-sonnet-20250219';
