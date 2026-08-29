-- Migrate deprecated Groq and OpenRouter models to their replacements
-- This updates all AgentNode blocks that use deprecated models that have been decommissioned
-- Deprecated models:
--   - deepseek-r1-distill-llama-70b (Groq - decommissioned)
--   - gemma2-9b-it (Groq - decommissioned)
--   - llama3-70b-8192 (Groq - decommissioned)
--   - llama3-8b-8192 (Groq - decommissioned)
--   - google/gemini-flash-1.5 (OpenRouter - no endpoints found)

-- Update llama3-70b-8192 to llama-3.3-70b-versatile
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"llama-3.3-70b-versatile"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' = 'llama3-70b-8192';

-- Update llama3-8b-8192 to llama-3.1-8b-instant
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"llama-3.1-8b-instant"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' = 'llama3-8b-8192';

-- Update google/gemini-flash-1.5 to google/gemini-2.5-flash
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"google/gemini-2.5-flash"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' = 'google/gemini-flash-1.5';

-- Update deepseek-r1-distill-llama-70b to gpt-5-chat-latest (no direct replacement)
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"gpt-5-chat-latest"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' = 'deepseek-r1-distill-llama-70b';

-- Update gemma2-9b-it to gpt-5-chat-latest (no direct replacement)
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"gpt-5-chat-latest"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' = 'gemma2-9b-it';
