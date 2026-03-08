-- Migration: Migrate deprecated Gemini 3 Pro Preview to Gemini 3.1 Pro Preview
-- Date: 2026-03-08
-- Reason: Google is shutting down Gemini 3 Pro Preview on March 9, 2026
-- Migration strategy: Replace all instances of "google/gemini-3-pro-preview" with "google/gemini-3.1-pro-preview"

-- Update AgentBlock input_default where model is set to the deprecated Gemini 3 Pro Preview
UPDATE "AgentBlock"
SET "input_default" = jsonb_set(
    "input_default",
    '{model}',
    '"google/gemini-3.1-pro-preview"'::jsonb
)
WHERE "input_default" -> 'model' = '"google/gemini-3-pro-preview"'::jsonb;

-- Update AgentBlock input_default for any nested model references (e.g., in complex configurations)
UPDATE "AgentBlock"
SET "input_default" = replace("input_default"::text, 'google/gemini-3-pro-preview', 'google/gemini-3.1-pro-preview')::jsonb
WHERE "input_default"::text LIKE '%google/gemini-3-pro-preview%';

-- Update AgentGraphExecution where any block uses the deprecated model
UPDATE "AgentGraphExecution"
SET "inputs" = replace("inputs"::text, 'google/gemini-3-pro-preview', 'google/gemini-3.1-pro-preview')::jsonb
WHERE "inputs"::text LIKE '%google/gemini-3-pro-preview%';

-- Update AgentNodeExecution where the deprecated model is referenced
UPDATE "AgentNodeExecution"
SET "executionData" = replace("executionData"::text, 'google/gemini-3-pro-preview', 'google/gemini-3.1-pro-preview')::jsonb
WHERE "executionData"::text LIKE '%google/gemini-3-pro-preview%';

-- Update any stored graphs that might have the deprecated model in their configuration
UPDATE "AgentGraph"
SET "graph_metadata" = replace("graph_metadata"::text, 'google/gemini-3-pro-preview', 'google/gemini-3.1-pro-preview')::jsonb
WHERE "graph_metadata"::text LIKE '%google/gemini-3-pro-preview%';

-- Update AgentNode constantInput field (CRITICAL: Prevents Pydantic validation errors when loading graphs)
UPDATE "AgentNode"
SET "constantInput" = replace("constantInput"::text, 'google/gemini-3-pro-preview', 'google/gemini-3.1-pro-preview')::jsonb
WHERE "constantInput"::text LIKE '%google/gemini-3-pro-preview%';

-- Log the migration completion
DO $$
BEGIN
    RAISE NOTICE 'Migration completed: Replaced all instances of google/gemini-3-pro-preview with google/gemini-3.1-pro-preview';
END $$;
