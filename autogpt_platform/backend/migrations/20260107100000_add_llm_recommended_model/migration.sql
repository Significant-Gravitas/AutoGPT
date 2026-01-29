-- Add isRecommended field to LlmModel table
-- This allows admins to mark a model as the recommended default

ALTER TABLE "LlmModel" ADD COLUMN "isRecommended" BOOLEAN NOT NULL DEFAULT false;

-- Set gpt-4o-mini as the default recommended model (if it exists)
UPDATE "LlmModel" SET "isRecommended" = true WHERE "slug" = 'gpt-4o-mini' AND "isEnabled" = true;

-- Create unique partial index to enforce only one recommended model at the database level
-- This prevents multiple rows from having isRecommended = true
CREATE UNIQUE INDEX "LlmModel_single_recommended_idx" ON "LlmModel" ("isRecommended") WHERE "isRecommended" = true;
