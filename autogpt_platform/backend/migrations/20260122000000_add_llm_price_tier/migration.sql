-- Add new columns to LlmModel table for extended model metadata
-- These columns support the LLM Picker UI enhancements

-- Add priceTier column: 1=cheapest, 2=medium, 3=expensive
ALTER TABLE "LlmModel" ADD COLUMN IF NOT EXISTS "priceTier" INTEGER NOT NULL DEFAULT 1;

-- Add creatorId column for model creator relationship (if not exists)
ALTER TABLE "LlmModel" ADD COLUMN IF NOT EXISTS "creatorId" TEXT;

-- Add isRecommended column (if not exists)
ALTER TABLE "LlmModel" ADD COLUMN IF NOT EXISTS "isRecommended" BOOLEAN NOT NULL DEFAULT FALSE;

-- Add index on creatorId if not exists
CREATE INDEX IF NOT EXISTS "LlmModel_creatorId_idx" ON "LlmModel"("creatorId");

-- Add foreign key for creatorId if not exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LlmModel_creatorId_fkey') THEN
        -- Only add FK if LlmModelCreator table exists
        IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'LlmModelCreator') THEN
            ALTER TABLE "LlmModel" ADD CONSTRAINT "LlmModel_creatorId_fkey"
            FOREIGN KEY ("creatorId") REFERENCES "LlmModelCreator"("id") ON DELETE SET NULL ON UPDATE CASCADE;
        END IF;
    END IF;
END $$;

-- Update priceTier values for existing models based on original MODEL_METADATA
-- Tier 1 = cheapest, Tier 2 = medium, Tier 3 = expensive

-- OpenAI models
UPDATE "LlmModel" SET "priceTier" = 2 WHERE "slug" = 'o3';
UPDATE "LlmModel" SET "priceTier" = 1 WHERE "slug" = 'o3-mini';
UPDATE "LlmModel" SET "priceTier" = 3 WHERE "slug" = 'o1';
UPDATE "LlmModel" SET "priceTier" = 2 WHERE "slug" = 'o1-mini';
UPDATE "LlmModel" SET "priceTier" = 3 WHERE "slug" = 'gpt-5.2';
UPDATE "LlmModel" SET "priceTier" = 2 WHERE "slug" = 'gpt-5.1';
UPDATE "LlmModel" SET "priceTier" = 1 WHERE "slug" = 'gpt-5';
UPDATE "LlmModel" SET "priceTier" = 1 WHERE "slug" = 'gpt-5-mini';
UPDATE "LlmModel" SET "priceTier" = 1 WHERE "slug" = 'gpt-5-nano';
UPDATE "LlmModel" SET "priceTier" = 2 WHERE "slug" = 'gpt-5-chat-latest';
UPDATE "LlmModel" SET "priceTier" = 1 WHERE "slug" LIKE 'gpt-4.1%';
UPDATE "LlmModel" SET "priceTier" = 1 WHERE "slug" = 'gpt-4o-mini';
UPDATE "LlmModel" SET "priceTier" = 2 WHERE "slug" = 'gpt-4o';
UPDATE "LlmModel" SET "priceTier" = 3 WHERE "slug" = 'gpt-4-turbo';
UPDATE "LlmModel" SET "priceTier" = 1 WHERE "slug" = 'gpt-3.5-turbo';

-- Anthropic models
UPDATE "LlmModel" SET "priceTier" = 3 WHERE "slug" LIKE 'claude-opus%';
UPDATE "LlmModel" SET "priceTier" = 2 WHERE "slug" LIKE 'claude-sonnet%';
UPDATE "LlmModel" SET "priceTier" = 3 WHERE "slug" LIKE 'claude%-4-5-sonnet%';
UPDATE "LlmModel" SET "priceTier" = 2 WHERE "slug" LIKE 'claude%-haiku%';
UPDATE "LlmModel" SET "priceTier" = 1 WHERE "slug" = 'claude-3-haiku-20240307';

-- OpenRouter models - Pro/expensive tiers
UPDATE "LlmModel" SET "priceTier" = 2 WHERE "slug" LIKE 'google/gemini%-pro%';
UPDATE "LlmModel" SET "priceTier" = 2 WHERE "slug" LIKE '%command-r-plus%';
UPDATE "LlmModel" SET "priceTier" = 2 WHERE "slug" LIKE '%sonar-pro%';
UPDATE "LlmModel" SET "priceTier" = 3 WHERE "slug" LIKE '%sonar-deep-research%';
UPDATE "LlmModel" SET "priceTier" = 3 WHERE "slug" = 'x-ai/grok-4';
UPDATE "LlmModel" SET "priceTier" = 3 WHERE "slug" LIKE '%qwen3-coder%';
