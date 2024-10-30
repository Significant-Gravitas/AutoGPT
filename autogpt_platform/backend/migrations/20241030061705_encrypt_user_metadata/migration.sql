-- First update all records to have empty JSON object
UPDATE "User"
SET "metadata" = '{}';

-- Then convert to TEXT and make required
ALTER TABLE "User"
ALTER COLUMN "metadata" TYPE TEXT USING metadata::TEXT,
ALTER COLUMN "metadata" SET DEFAULT '',
ALTER COLUMN "metadata" SET NOT NULL;

-- Finally set all to empty string if you want to clear them
UPDATE "User"
SET "metadata" = '';