-- Fix PerplexityBlock nodes that have invalid model values (e.g. gpt-4o,
-- gpt-5.2-2025-12-11) set by the agent generator. Defaults them to the
-- standard "perplexity/sonar" model.
--
-- PerplexityBlock ID: c8a5f2e9-8b3d-4a7e-9f6c-1d5e3c9b7a4f
-- Valid models: perplexity/sonar, perplexity/sonar-pro, perplexity/sonar-deep-research

UPDATE "AgentNode"
SET "constantInput" = JSONB_SET(
    "constantInput"::jsonb,
    '{model}',
    '"perplexity/sonar"'::jsonb
)
WHERE "agentBlockId" = 'c8a5f2e9-8b3d-4a7e-9f6c-1d5e3c9b7a4f'
  AND "constantInput"::jsonb ? 'model'
  AND "constantInput"::jsonb->>'model' NOT IN (
      'perplexity/sonar',
      'perplexity/sonar-pro',
      'perplexity/sonar-deep-research'
  );

-- Update AgentPreset input overrides (stored in AgentNodeExecutionInputOutput).
-- The table links to AgentNode through AgentNodeExecution, not directly.
UPDATE "AgentNodeExecutionInputOutput" io
SET "data" = JSONB_SET(
    io."data"::jsonb,
    '{model}',
    '"perplexity/sonar"'::jsonb
)
FROM "AgentNodeExecution" exe
JOIN "AgentNode" n ON n."id" = exe."agentNodeId"
WHERE io."agentPresetId" IS NOT NULL
  AND (io."referencedByInputExecId" = exe."id" OR io."referencedByOutputExecId" = exe."id")
  AND n."agentBlockId" = 'c8a5f2e9-8b3d-4a7e-9f6c-1d5e3c9b7a4f'
  AND io."data"::jsonb ? 'model'
  AND io."data"::jsonb->>'model' NOT IN (
      'perplexity/sonar',
      'perplexity/sonar-pro',
      'perplexity/sonar-deep-research'
  );
