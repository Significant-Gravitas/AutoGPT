-- Add composite index on LlmModelMigration for optimized active migration queries
-- This index improves performance when querying for non-reverted migrations by model slug
-- Used by the billing system to apply customCreditCost overrides

-- CreateIndex
CREATE INDEX "LlmModelMigration_sourceModelSlug_isReverted_idx" ON "LlmModelMigration"("sourceModelSlug", "isReverted");
