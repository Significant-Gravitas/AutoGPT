-- Move capability fields from LlmProvider to LlmModel
-- Per reviewer feedback: capabilities should be per-model, not per-provider
-- (e.g., Hugging Face hosts models with vastly different capabilities)

-- Add capability columns to LlmModel
ALTER TABLE platform."LlmModel" ADD COLUMN "supportsTools" BOOLEAN NOT NULL DEFAULT true;
ALTER TABLE platform."LlmModel" ADD COLUMN "supportsJsonOutput" BOOLEAN NOT NULL DEFAULT true;
ALTER TABLE platform."LlmModel" ADD COLUMN "supportsReasoning" BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE platform."LlmModel" ADD COLUMN "supportsParallelTool" BOOLEAN NOT NULL DEFAULT false;

-- Copy capability values from provider to all its models
UPDATE platform."LlmModel" m
SET 
  "supportsTools" = p."supportsTools",
  "supportsJsonOutput" = p."supportsJsonOutput",
  "supportsReasoning" = p."supportsReasoning",
  "supportsParallelTool" = p."supportsParallelTool"
FROM platform."LlmProvider" p
WHERE m."providerId" = p.id;

-- Remove capability columns from LlmProvider
ALTER TABLE platform."LlmProvider" DROP COLUMN "supportsTools";
ALTER TABLE platform."LlmProvider" DROP COLUMN "supportsJsonOutput";
ALTER TABLE platform."LlmProvider" DROP COLUMN "supportsReasoning";
ALTER TABLE platform."LlmProvider" DROP COLUMN "supportsParallelTool";
