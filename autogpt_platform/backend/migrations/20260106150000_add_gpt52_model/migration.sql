-- Add GPT-5.2 model and update O3 slug
-- This migration adds the new GPT-5.2 model added in dev branch

-- Update O3 slug to match dev branch format
UPDATE "LlmModel"
SET "slug" = 'o3-2025-04-16'
WHERE "slug" = 'o3';

-- Update cost reference for O3 if needed
-- (costs are linked by model ID, so no update needed)

-- Add GPT-5.2 model
WITH provider_id AS (
    SELECT "id" FROM "LlmProvider" WHERE "name" = 'openai'
)
INSERT INTO "LlmModel" ("id", "slug", "displayName", "description", "providerId", "contextWindow", "maxOutputTokens", "isEnabled", "capabilities", "metadata")
SELECT
    gen_random_uuid(),
    'gpt-5.2-2025-12-11',
    'GPT 5.2',
    'OpenAI GPT-5.2 model',
    p."id",
    400000,
    128000,
    true,
    '{}'::jsonb,
    '{}'::jsonb
FROM provider_id p
ON CONFLICT ("slug") DO NOTHING;

-- Add cost for GPT-5.2
WITH model_id AS (
    SELECT m."id", p."name" as provider_name
    FROM "LlmModel" m
    JOIN "LlmProvider" p ON p."id" = m."providerId"
    WHERE m."slug" = 'gpt-5.2-2025-12-11'
)
INSERT INTO "LlmModelCost" ("id", "unit", "creditCost", "credentialProvider", "credentialId", "credentialType", "currency", "metadata", "llmModelId")
SELECT
    gen_random_uuid(),
    'RUN'::"LlmCostUnit",
    3,  -- Same cost tier as GPT-5.1
    m.provider_name,
    NULL,
    'api_key',
    NULL,
    '{}'::jsonb,
    m."id"
FROM model_id m
WHERE NOT EXISTS (
    SELECT 1 FROM "LlmModelCost" c WHERE c."llmModelId" = m."id"
);
