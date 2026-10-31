-- Retire the undated aliases of Claude Opus 4.1, Opus 4 and Sonnet 4, which
-- Anthropic retired with their snapshots, plus their anthropic/-prefixed and
-- OpenRouter spellings. Same map as LEGACY_MODEL_MAPPINGS; the boot-time
-- `migrate_llm_models` applies it too.

-- Opus 4.1 and Opus 4 aliases -> claude-opus-4-7
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"claude-opus-4-7"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' IN (
         'claude-opus-4-1',
         'anthropic/claude-opus-4-1',
         'anthropic/claude-opus-4.1',
         'claude-opus-4-0',
         'anthropic/claude-opus-4-0',
         'anthropic/claude-opus-4'
       );

UPDATE "AgentNodeExecutionInputOutput"
SET    "data" = JSONB_SET(
         "data"::jsonb,
         '{model}',
         '"claude-opus-4-7"'::jsonb
       )
WHERE  "agentPresetId" IS NOT NULL
  AND  "data"::jsonb->>'model' IN (
         'claude-opus-4-1',
         'anthropic/claude-opus-4-1',
         'anthropic/claude-opus-4.1',
         'claude-opus-4-0',
         'anthropic/claude-opus-4-0',
         'anthropic/claude-opus-4'
       );

-- Sonnet 4 aliases -> claude-sonnet-4-6
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{model}',
         '"claude-sonnet-4-6"'::jsonb
       )
WHERE  "constantInput"::jsonb->>'model' IN (
         'claude-sonnet-4-0',
         'anthropic/claude-sonnet-4-0',
         'anthropic/claude-sonnet-4'
       );

UPDATE "AgentNodeExecutionInputOutput"
SET    "data" = JSONB_SET(
         "data"::jsonb,
         '{model}',
         '"claude-sonnet-4-6"'::jsonb
       )
WHERE  "agentPresetId" IS NOT NULL
  AND  "data"::jsonb->>'model' IN (
         'claude-sonnet-4-0',
         'anthropic/claude-sonnet-4-0',
         'anthropic/claude-sonnet-4'
       );
