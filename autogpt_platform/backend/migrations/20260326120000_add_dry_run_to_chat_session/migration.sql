-- Replace dryRun Boolean column with extensible metadata Json column.
-- Migrate existing dryRun values into the new metadata JSON.

-- Step 1: Add the new metadata column with a default empty JSON object.
ALTER TABLE "ChatSession" ADD COLUMN "metadata" JSONB NOT NULL DEFAULT '{}';

-- Step 2: Migrate existing dryRun=true rows into the metadata column.
UPDATE "ChatSession"
SET "metadata" = jsonb_build_object('dry_run', true)
WHERE "dryRun" = true;

-- Step 3: Drop the old dryRun column.
ALTER TABLE "ChatSession" DROP COLUMN "dryRun";
