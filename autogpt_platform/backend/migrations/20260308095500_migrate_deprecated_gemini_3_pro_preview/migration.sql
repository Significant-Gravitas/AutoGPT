-- Migrate Gemini 3 Pro Preview to Gemini 3.1 Pro Preview
-- This updates all AgentNode blocks that use the deprecated Gemini 3 Pro Preview model
-- Google is shutting down google/gemini-3-pro-preview on March 9, 2026

-- Update AgentNode constant inputs
UPDATE "AgentNode"
SET "constantInput" = JSONB_SET(
    "constantInput"::jsonb,
    '{model}',
    '"google/gemini-3.1-pro-preview"'::jsonb
)
WHERE "constantInput"::jsonb->>'model' = 'google/gemini-3-pro-preview';

-- Update AgentPreset input overrides (stored in AgentNodeExecutionInputOutput)
UPDATE "AgentNodeExecutionInputOutput"
SET "data" = JSONB_SET(
    "data"::jsonb,
    '{model}',
    '"google/gemini-3.1-pro-preview"'::jsonb
)
WHERE "agentPresetId" IS NOT NULL
  AND "data"::jsonb->>'model' = 'google/gemini-3-pro-preview';
