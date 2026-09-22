-- Retire deprecated LLM models with family-aware migration.
--
-- Each block updates two tables:
--   * "AgentNode"."constantInput" — saved graph definitions.
--   * "AgentNodeExecutionInputOutput"."data" where "agentPresetId" IS NOT NULL —
--     preset overrides.
--
-- Mappings are family-aware: Claude Opus → newer Opus, Claude Sonnet → newer
-- Sonnet, Gemini Flash → newer Flash, etc. The boot-time safety net
-- (`migrate_llm_models` in backend/data/graph.py) applies the same mapping so
-- environments stay consistent even before this migration runs.
--
-- For bare Anthropic/OpenAI slugs we also match the provider-prefixed form
-- (e.g. `anthropic/claude-sonnet-4-20250514`) because `LlmModel._missing_`
-- accepts prefixed inputs at write time, so historical rows may carry either
-- form.
--
-- AI/ML API stragglers (Qwen/Qwen2.5-72B-Instruct-Turbo,
-- nvidia/llama-3.1-nemotron-70b-instruct,
-- meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo,
-- meta-llama/Llama-3.2-3B-Instruct-Turbo) have no direct same-family successor
-- on AI/ML's current catalogue, so they all map to
-- meta-llama/Llama-3.3-70B-Instruct-Turbo, which AI/ML still serves and is the
-- closest open-weight Meta/Llama generation.

-- claude-3-haiku-20240307 -> claude-haiku-4-5-20251001
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"claude-haiku-4-5-20251001"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' IN (
         'claude-3-haiku-20240307',
         'anthropic/claude-3-haiku-20240307'
       );

UPDATE "AgentNodeExecutionInputOutput"
SET    "data" = JSONB_SET(
         "data"::jsonb,
         '{model}',
         '"claude-haiku-4-5-20251001"'::jsonb
       )
WHERE  "agentPresetId" IS NOT NULL
  AND  "data"::jsonb->>'model' IN (
         'claude-3-haiku-20240307',
         'anthropic/claude-3-haiku-20240307'
       );

-- claude-opus-4-20250514 -> claude-opus-4-7
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"claude-opus-4-7"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' IN (
         'claude-opus-4-20250514',
         'anthropic/claude-opus-4-20250514'
       );

UPDATE "AgentNodeExecutionInputOutput"
SET    "data" = JSONB_SET(
         "data"::jsonb,
         '{model}',
         '"claude-opus-4-7"'::jsonb
       )
WHERE  "agentPresetId" IS NOT NULL
  AND  "data"::jsonb->>'model' IN (
         'claude-opus-4-20250514',
         'anthropic/claude-opus-4-20250514'
       );

-- claude-sonnet-4-20250514 -> claude-sonnet-4-6
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"claude-sonnet-4-6"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' IN (
         'claude-sonnet-4-20250514',
         'anthropic/claude-sonnet-4-20250514'
       );

UPDATE "AgentNodeExecutionInputOutput"
SET    "data" = JSONB_SET(
         "data"::jsonb,
         '{model}',
         '"claude-sonnet-4-6"'::jsonb
       )
WHERE  "agentPresetId" IS NOT NULL
  AND  "data"::jsonb->>'model' IN (
         'claude-sonnet-4-20250514',
         'anthropic/claude-sonnet-4-20250514'
       );

-- claude-opus-4-1-20250805 -> claude-opus-4-7
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"claude-opus-4-7"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' IN (
         'claude-opus-4-1-20250805',
         'anthropic/claude-opus-4-1-20250805'
       );

UPDATE "AgentNodeExecutionInputOutput"
SET    "data" = JSONB_SET(
         "data"::jsonb,
         '{model}',
         '"claude-opus-4-7"'::jsonb
       )
WHERE  "agentPresetId" IS NOT NULL
  AND  "data"::jsonb->>'model' IN (
         'claude-opus-4-1-20250805',
         'anthropic/claude-opus-4-1-20250805'
       );

-- gpt-4-turbo -> gpt-4.1-2025-04-14
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"gpt-4.1-2025-04-14"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' IN (
         'gpt-4-turbo',
         'openai/gpt-4-turbo'
       );

UPDATE "AgentNodeExecutionInputOutput"
SET    "data" = JSONB_SET(
         "data"::jsonb,
         '{model}',
         '"gpt-4.1-2025-04-14"'::jsonb
       )
WHERE  "agentPresetId" IS NOT NULL
  AND  "data"::jsonb->>'model' IN (
         'gpt-4-turbo',
         'openai/gpt-4-turbo'
       );

-- o1 -> o3-2025-04-16
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"o3-2025-04-16"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' IN (
         'o1',
         'openai/o1'
       );

UPDATE "AgentNodeExecutionInputOutput"
SET    "data" = JSONB_SET(
         "data"::jsonb,
         '{model}',
         '"o3-2025-04-16"'::jsonb
       )
WHERE  "agentPresetId" IS NOT NULL
  AND  "data"::jsonb->>'model' IN (
         'o1',
         'openai/o1'
       );

-- o1-mini -> o3-mini
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"o3-mini"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' IN (
         'o1-mini',
         'openai/o1-mini'
       );

UPDATE "AgentNodeExecutionInputOutput"
SET    "data" = JSONB_SET(
         "data"::jsonb,
         '{model}',
         '"o3-mini"'::jsonb
       )
WHERE  "agentPresetId" IS NOT NULL
  AND  "data"::jsonb->>'model' IN (
         'o1-mini',
         'openai/o1-mini'
       );

-- google/gemini-2.5-pro-preview-03-25 -> google/gemini-2.5-pro
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"google/gemini-2.5-pro"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' = 'google/gemini-2.5-pro-preview-03-25';

UPDATE "AgentNodeExecutionInputOutput"
SET    "data" = JSONB_SET(
         "data"::jsonb,
         '{model}',
         '"google/gemini-2.5-pro"'::jsonb
       )
WHERE  "agentPresetId" IS NOT NULL
  AND  "data"::jsonb->>'model' = 'google/gemini-2.5-pro-preview-03-25';

-- google/gemini-2.5-flash-lite-preview-06-17 -> google/gemini-2.5-flash
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"google/gemini-2.5-flash"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' = 'google/gemini-2.5-flash-lite-preview-06-17';

UPDATE "AgentNodeExecutionInputOutput"
SET    "data" = JSONB_SET(
         "data"::jsonb,
         '{model}',
         '"google/gemini-2.5-flash"'::jsonb
       )
WHERE  "agentPresetId" IS NOT NULL
  AND  "data"::jsonb->>'model' = 'google/gemini-2.5-flash-lite-preview-06-17';

-- cohere/command-r-08-2024 -> cohere/command-a-03-2025
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"cohere/command-a-03-2025"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' = 'cohere/command-r-08-2024';

UPDATE "AgentNodeExecutionInputOutput"
SET    "data" = JSONB_SET(
         "data"::jsonb,
         '{model}',
         '"cohere/command-a-03-2025"'::jsonb
       )
WHERE  "agentPresetId" IS NOT NULL
  AND  "data"::jsonb->>'model' = 'cohere/command-r-08-2024';

-- cohere/command-r-plus-08-2024 -> cohere/command-a-03-2025
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"cohere/command-a-03-2025"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' = 'cohere/command-r-plus-08-2024';

UPDATE "AgentNodeExecutionInputOutput"
SET    "data" = JSONB_SET(
         "data"::jsonb,
         '{model}',
         '"cohere/command-a-03-2025"'::jsonb
       )
WHERE  "agentPresetId" IS NOT NULL
  AND  "data"::jsonb->>'model' = 'cohere/command-r-plus-08-2024';

-- mistralai/mistral-nemo -> mistralai/mistral-small-3.2-24b-instruct
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"mistralai/mistral-small-3.2-24b-instruct"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' = 'mistralai/mistral-nemo';

UPDATE "AgentNodeExecutionInputOutput"
SET    "data" = JSONB_SET(
         "data"::jsonb,
         '{model}',
         '"mistralai/mistral-small-3.2-24b-instruct"'::jsonb
       )
WHERE  "agentPresetId" IS NOT NULL
  AND  "data"::jsonb->>'model' = 'mistralai/mistral-nemo';

-- microsoft/wizardlm-2-8x22b -> microsoft/phi-4
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"microsoft/phi-4"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' = 'microsoft/wizardlm-2-8x22b';

UPDATE "AgentNodeExecutionInputOutput"
SET    "data" = JSONB_SET(
         "data"::jsonb,
         '{model}',
         '"microsoft/phi-4"'::jsonb
       )
WHERE  "agentPresetId" IS NOT NULL
  AND  "data"::jsonb->>'model' = 'microsoft/wizardlm-2-8x22b';

-- moonshotai/kimi-k2 -> moonshotai/kimi-k2.6
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"moonshotai/kimi-k2.6"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' = 'moonshotai/kimi-k2';

UPDATE "AgentNodeExecutionInputOutput"
SET    "data" = JSONB_SET(
         "data"::jsonb,
         '{model}',
         '"moonshotai/kimi-k2.6"'::jsonb
       )
WHERE  "agentPresetId" IS NOT NULL
  AND  "data"::jsonb->>'model' = 'moonshotai/kimi-k2';

-- moonshotai/kimi-k2-0905 -> moonshotai/kimi-k2.6
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"moonshotai/kimi-k2.6"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' = 'moonshotai/kimi-k2-0905';

UPDATE "AgentNodeExecutionInputOutput"
SET    "data" = JSONB_SET(
         "data"::jsonb,
         '{model}',
         '"moonshotai/kimi-k2.6"'::jsonb
       )
WHERE  "agentPresetId" IS NOT NULL
  AND  "data"::jsonb->>'model' = 'moonshotai/kimi-k2-0905';

-- z-ai/glm-4-32b -> z-ai/glm-4.6
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"z-ai/glm-4.6"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' = 'z-ai/glm-4-32b';

UPDATE "AgentNodeExecutionInputOutput"
SET    "data" = JSONB_SET(
         "data"::jsonb,
         '{model}',
         '"z-ai/glm-4.6"'::jsonb
       )
WHERE  "agentPresetId" IS NOT NULL
  AND  "data"::jsonb->>'model' = 'z-ai/glm-4-32b';

-- z-ai/glm-4.5 -> z-ai/glm-4.6
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"z-ai/glm-4.6"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' = 'z-ai/glm-4.5';

UPDATE "AgentNodeExecutionInputOutput"
SET    "data" = JSONB_SET(
         "data"::jsonb,
         '{model}',
         '"z-ai/glm-4.6"'::jsonb
       )
WHERE  "agentPresetId" IS NOT NULL
  AND  "data"::jsonb->>'model' = 'z-ai/glm-4.5';

-- z-ai/glm-4.5-air -> z-ai/glm-4.7-flash
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"z-ai/glm-4.7-flash"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' = 'z-ai/glm-4.5-air';

UPDATE "AgentNodeExecutionInputOutput"
SET    "data" = JSONB_SET(
         "data"::jsonb,
         '{model}',
         '"z-ai/glm-4.7-flash"'::jsonb
       )
WHERE  "agentPresetId" IS NOT NULL
  AND  "data"::jsonb->>'model' = 'z-ai/glm-4.5-air';

-- z-ai/glm-4.5-air:free -> z-ai/glm-4.7-flash
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"z-ai/glm-4.7-flash"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' = 'z-ai/glm-4.5-air:free';

UPDATE "AgentNodeExecutionInputOutput"
SET    "data" = JSONB_SET(
         "data"::jsonb,
         '{model}',
         '"z-ai/glm-4.7-flash"'::jsonb
       )
WHERE  "agentPresetId" IS NOT NULL
  AND  "data"::jsonb->>'model' = 'z-ai/glm-4.5-air:free';

-- z-ai/glm-4.5v -> z-ai/glm-4.6v
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"z-ai/glm-4.6v"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' = 'z-ai/glm-4.5v';

UPDATE "AgentNodeExecutionInputOutput"
SET    "data" = JSONB_SET(
         "data"::jsonb,
         '{model}',
         '"z-ai/glm-4.6v"'::jsonb
       )
WHERE  "agentPresetId" IS NOT NULL
  AND  "data"::jsonb->>'model' = 'z-ai/glm-4.5v';

-- Qwen/Qwen2.5-72B-Instruct-Turbo -> meta-llama/Llama-3.3-70B-Instruct-Turbo
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"meta-llama/Llama-3.3-70B-Instruct-Turbo"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' = 'Qwen/Qwen2.5-72B-Instruct-Turbo';

UPDATE "AgentNodeExecutionInputOutput"
SET    "data" = JSONB_SET(
         "data"::jsonb,
         '{model}',
         '"meta-llama/Llama-3.3-70B-Instruct-Turbo"'::jsonb
       )
WHERE  "agentPresetId" IS NOT NULL
  AND  "data"::jsonb->>'model' = 'Qwen/Qwen2.5-72B-Instruct-Turbo';

-- nvidia/llama-3.1-nemotron-70b-instruct -> meta-llama/Llama-3.3-70B-Instruct-Turbo
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"meta-llama/Llama-3.3-70B-Instruct-Turbo"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' = 'nvidia/llama-3.1-nemotron-70b-instruct';

UPDATE "AgentNodeExecutionInputOutput"
SET    "data" = JSONB_SET(
         "data"::jsonb,
         '{model}',
         '"meta-llama/Llama-3.3-70B-Instruct-Turbo"'::jsonb
       )
WHERE  "agentPresetId" IS NOT NULL
  AND  "data"::jsonb->>'model' = 'nvidia/llama-3.1-nemotron-70b-instruct';

-- meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo -> meta-llama/Llama-3.3-70B-Instruct-Turbo
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"meta-llama/Llama-3.3-70B-Instruct-Turbo"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' = 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo';

UPDATE "AgentNodeExecutionInputOutput"
SET    "data" = JSONB_SET(
         "data"::jsonb,
         '{model}',
         '"meta-llama/Llama-3.3-70B-Instruct-Turbo"'::jsonb
       )
WHERE  "agentPresetId" IS NOT NULL
  AND  "data"::jsonb->>'model' = 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo';

-- meta-llama/Llama-3.2-3B-Instruct-Turbo -> meta-llama/Llama-3.3-70B-Instruct-Turbo
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"meta-llama/Llama-3.3-70B-Instruct-Turbo"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' = 'meta-llama/Llama-3.2-3B-Instruct-Turbo';

UPDATE "AgentNodeExecutionInputOutput"
SET    "data" = JSONB_SET(
         "data"::jsonb,
         '{model}',
         '"meta-llama/Llama-3.3-70B-Instruct-Turbo"'::jsonb
       )
WHERE  "agentPresetId" IS NOT NULL
  AND  "data"::jsonb->>'model' = 'meta-llama/Llama-3.2-3B-Instruct-Turbo';
