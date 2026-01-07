-- Add isRecommended field to LlmModel table
-- This allows admins to mark a model as the recommended default

ALTER TABLE "LlmModel" ADD COLUMN "isRecommended" BOOLEAN NOT NULL DEFAULT false;

-- Set gpt-4o-mini as the default recommended model (if it exists)
UPDATE "LlmModel" SET "isRecommended" = true WHERE "slug" = 'gpt-4o-mini' AND "isEnabled" = true;

-- Create index for quick lookup of recommended model
CREATE INDEX "LlmModel_isRecommended_idx" ON "LlmModel" ("isRecommended") WHERE "isRecommended" = true;
