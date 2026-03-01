-- Sync LLM models with latest dev branch changes
-- This migration adds new models and removes deprecated ones

-- Remove models that were deleted from dev
DELETE FROM "LlmModelCost" WHERE "llmModelId" IN (
    SELECT "id" FROM "LlmModel" WHERE "slug" IN ('o3', 'o3-mini', 'claude-3-7-sonnet-20250219')
);

DELETE FROM "LlmModel" WHERE "slug" IN ('o3', 'o3-mini', 'claude-3-7-sonnet-20250219');

-- Add new models from dev
WITH provider_ids AS (
    SELECT "id", "name" FROM "LlmProvider"
)
INSERT INTO "LlmModel" ("id", "slug", "displayName", "description", "providerId", "contextWindow", "maxOutputTokens", "isEnabled", "capabilities", "metadata", "createdAt", "updatedAt")
SELECT
    gen_random_uuid(),
    model_slug,
    model_display_name,
    NULL,
    p."id",
    context_window,
    max_output_tokens,
    true,
    '{}'::jsonb,
    '{}'::jsonb,
    NOW(),
    NOW()
FROM (VALUES
    -- New OpenAI model
    ('gpt-5.2-2025-12-11', 'GPT 5.2', 'openai', 400000, 128000),
    -- New Anthropic model
    ('claude-opus-4-6', 'Claude 4.6 Opus', 'anthropic', 200000, 64000)
) AS models(model_slug, model_display_name, provider_name, context_window, max_output_tokens)
JOIN provider_ids p ON p."name" = models.provider_name
ON CONFLICT ("slug") DO NOTHING;

-- Add costs for new models
WITH model_ids AS (
    SELECT "id", "slug", "providerId" FROM "LlmModel"
),
provider_ids AS (
    SELECT "id", "name" FROM "LlmProvider"
)
INSERT INTO "LlmModelCost" ("id", "unit", "creditCost", "credentialProvider", "credentialId", "credentialType", "currency", "metadata", "llmModelId", "createdAt", "updatedAt")
SELECT
    gen_random_uuid(),
    'RUN'::"LlmCostUnit",
    cost,
    p."name",
    NULL,
    'api_key',
    NULL,
    '{}'::jsonb,
    m."id",
    NOW(),
    NOW()
FROM (VALUES
    -- New model costs (estimate based on similar models)
    ('gpt-5.2-2025-12-11', 5),  -- Similar to GPT 5.1
    ('claude-opus-4-6', 21)     -- Similar to other Opus 4.x models
) AS costs(model_slug, cost)
JOIN model_ids m ON m."slug" = costs.model_slug
JOIN provider_ids p ON p."id" = m."providerId"
ON CONFLICT ("llmModelId", "credentialProvider", "unit") DO NOTHING;
