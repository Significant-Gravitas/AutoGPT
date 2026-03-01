-- Add LlmModelCreator table
-- Creator represents who made/trained the model (e.g., OpenAI, Meta)
-- This is distinct from Provider who hosts/serves the model (e.g., OpenRouter)

-- Create the LlmModelCreator table
CREATE TABLE "LlmModelCreator" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "name" TEXT NOT NULL,
    "displayName" TEXT NOT NULL,
    "description" TEXT,
    "websiteUrl" TEXT,
    "logoUrl" TEXT,
    "metadata" JSONB NOT NULL DEFAULT '{}',

    CONSTRAINT "LlmModelCreator_pkey" PRIMARY KEY ("id")
);

-- Create unique index on name
CREATE UNIQUE INDEX "LlmModelCreator_name_key" ON "LlmModelCreator"("name");

-- Add creatorId column to LlmModel
ALTER TABLE "LlmModel" ADD COLUMN "creatorId" TEXT;

-- Add foreign key constraint
ALTER TABLE "LlmModel" ADD CONSTRAINT "LlmModel_creatorId_fkey"
    FOREIGN KEY ("creatorId") REFERENCES "LlmModelCreator"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- Create index on creatorId
CREATE INDEX "LlmModel_creatorId_idx" ON "LlmModel"("creatorId");

-- Seed creators based on known model creators
INSERT INTO "LlmModelCreator" ("id", "updatedAt", "name", "displayName", "description", "websiteUrl", "metadata")
VALUES
    (gen_random_uuid(), CURRENT_TIMESTAMP, 'openai', 'OpenAI', 'Creator of GPT models', 'https://openai.com', '{}'),
    (gen_random_uuid(), CURRENT_TIMESTAMP, 'anthropic', 'Anthropic', 'Creator of Claude models', 'https://anthropic.com', '{}'),
    (gen_random_uuid(), CURRENT_TIMESTAMP, 'meta', 'Meta', 'Creator of Llama models', 'https://ai.meta.com', '{}'),
    (gen_random_uuid(), CURRENT_TIMESTAMP, 'google', 'Google', 'Creator of Gemini models', 'https://deepmind.google', '{}'),
    (gen_random_uuid(), CURRENT_TIMESTAMP, 'mistral', 'Mistral AI', 'Creator of Mistral models', 'https://mistral.ai', '{}'),
    (gen_random_uuid(), CURRENT_TIMESTAMP, 'cohere', 'Cohere', 'Creator of Command models', 'https://cohere.com', '{}'),
    (gen_random_uuid(), CURRENT_TIMESTAMP, 'deepseek', 'DeepSeek', 'Creator of DeepSeek models', 'https://deepseek.com', '{}'),
    (gen_random_uuid(), CURRENT_TIMESTAMP, 'perplexity', 'Perplexity AI', 'Creator of Sonar models', 'https://perplexity.ai', '{}'),
    (gen_random_uuid(), CURRENT_TIMESTAMP, 'qwen', 'Qwen (Alibaba)', 'Creator of Qwen models', 'https://qwenlm.github.io', '{}'),
    (gen_random_uuid(), CURRENT_TIMESTAMP, 'xai', 'xAI', 'Creator of Grok models', 'https://x.ai', '{}'),
    (gen_random_uuid(), CURRENT_TIMESTAMP, 'amazon', 'Amazon', 'Creator of Nova models', 'https://aws.amazon.com/bedrock', '{}'),
    (gen_random_uuid(), CURRENT_TIMESTAMP, 'microsoft', 'Microsoft', 'Creator of WizardLM models', 'https://microsoft.com', '{}'),
    (gen_random_uuid(), CURRENT_TIMESTAMP, 'moonshot', 'Moonshot AI', 'Creator of Kimi models', 'https://moonshot.cn', '{}'),
    (gen_random_uuid(), CURRENT_TIMESTAMP, 'nvidia', 'NVIDIA', 'Creator of Nemotron models', 'https://nvidia.com', '{}'),
    (gen_random_uuid(), CURRENT_TIMESTAMP, 'nous_research', 'Nous Research', 'Creator of Hermes models', 'https://nousresearch.com', '{}'),
    (gen_random_uuid(), CURRENT_TIMESTAMP, 'vercel', 'Vercel', 'Creator of v0 models', 'https://vercel.com', '{}'),
    (gen_random_uuid(), CURRENT_TIMESTAMP, 'cognitive_computations', 'Cognitive Computations', 'Creator of Dolphin models', 'https://erichartford.com', '{}'),
    (gen_random_uuid(), CURRENT_TIMESTAMP, 'gryphe', 'Gryphe', 'Creator of MythoMax models', 'https://huggingface.co/Gryphe', '{}')
ON CONFLICT ("name") DO NOTHING;

-- Update existing models with their creators
-- OpenAI models
UPDATE "LlmModel" SET "creatorId" = (SELECT "id" FROM "LlmModelCreator" WHERE "name" = 'openai')
WHERE "slug" LIKE 'gpt-%' OR "slug" LIKE 'o1%' OR "slug" LIKE 'o3%' OR "slug" LIKE 'openai/%';

-- Anthropic models
UPDATE "LlmModel" SET "creatorId" = (SELECT "id" FROM "LlmModelCreator" WHERE "name" = 'anthropic')
WHERE "slug" LIKE 'claude-%';

-- Meta/Llama models
UPDATE "LlmModel" SET "creatorId" = (SELECT "id" FROM "LlmModelCreator" WHERE "name" = 'meta')
WHERE "slug" LIKE 'llama%' OR "slug" LIKE 'Llama%' OR "slug" LIKE 'meta-llama/%' OR "slug" LIKE '%/llama-%';

-- Google models
UPDATE "LlmModel" SET "creatorId" = (SELECT "id" FROM "LlmModelCreator" WHERE "name" = 'google')
WHERE "slug" LIKE 'google/%' OR "slug" LIKE 'gemini%';

-- Mistral models
UPDATE "LlmModel" SET "creatorId" = (SELECT "id" FROM "LlmModelCreator" WHERE "name" = 'mistral')
WHERE "slug" LIKE 'mistral%' OR "slug" LIKE 'mistralai/%';

-- Cohere models
UPDATE "LlmModel" SET "creatorId" = (SELECT "id" FROM "LlmModelCreator" WHERE "name" = 'cohere')
WHERE "slug" LIKE 'cohere/%' OR "slug" LIKE 'command-%';

-- DeepSeek models
UPDATE "LlmModel" SET "creatorId" = (SELECT "id" FROM "LlmModelCreator" WHERE "name" = 'deepseek')
WHERE "slug" LIKE 'deepseek/%' OR "slug" LIKE 'deepseek-%';

-- Perplexity models
UPDATE "LlmModel" SET "creatorId" = (SELECT "id" FROM "LlmModelCreator" WHERE "name" = 'perplexity')
WHERE "slug" LIKE 'perplexity/%' OR "slug" LIKE 'sonar%';

-- Qwen models
UPDATE "LlmModel" SET "creatorId" = (SELECT "id" FROM "LlmModelCreator" WHERE "name" = 'qwen')
WHERE "slug" LIKE 'Qwen/%' OR "slug" LIKE 'qwen/%';

-- xAI/Grok models
UPDATE "LlmModel" SET "creatorId" = (SELECT "id" FROM "LlmModelCreator" WHERE "name" = 'xai')
WHERE "slug" LIKE 'x-ai/%' OR "slug" LIKE 'grok%';

-- Amazon models
UPDATE "LlmModel" SET "creatorId" = (SELECT "id" FROM "LlmModelCreator" WHERE "name" = 'amazon')
WHERE "slug" LIKE 'amazon/%' OR "slug" LIKE 'nova-%';

-- Microsoft models
UPDATE "LlmModel" SET "creatorId" = (SELECT "id" FROM "LlmModelCreator" WHERE "name" = 'microsoft')
WHERE "slug" LIKE 'microsoft/%' OR "slug" LIKE 'wizardlm%';

-- Moonshot models
UPDATE "LlmModel" SET "creatorId" = (SELECT "id" FROM "LlmModelCreator" WHERE "name" = 'moonshot')
WHERE "slug" LIKE 'moonshotai/%' OR "slug" LIKE 'kimi%';

-- NVIDIA models
UPDATE "LlmModel" SET "creatorId" = (SELECT "id" FROM "LlmModelCreator" WHERE "name" = 'nvidia')
WHERE "slug" LIKE 'nvidia/%' OR "slug" LIKE '%nemotron%';

-- Nous Research models
UPDATE "LlmModel" SET "creatorId" = (SELECT "id" FROM "LlmModelCreator" WHERE "name" = 'nous_research')
WHERE "slug" LIKE 'nousresearch/%' OR "slug" LIKE 'hermes%';

-- Vercel/v0 models
UPDATE "LlmModel" SET "creatorId" = (SELECT "id" FROM "LlmModelCreator" WHERE "name" = 'vercel')
WHERE "slug" LIKE 'v0-%';

-- Dolphin models (Cognitive Computations / Eric Hartford)
UPDATE "LlmModel" SET "creatorId" = (SELECT "id" FROM "LlmModelCreator" WHERE "name" = 'cognitive_computations')
WHERE "slug" LIKE 'dolphin-%';

-- Gryphe models
UPDATE "LlmModel" SET "creatorId" = (SELECT "id" FROM "LlmModelCreator" WHERE "name" = 'gryphe')
WHERE "slug" LIKE 'gryphe/%' OR "slug" LIKE 'mythomax%';
