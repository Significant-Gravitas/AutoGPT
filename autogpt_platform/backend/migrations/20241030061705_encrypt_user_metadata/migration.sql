-- Make User.metadata column consistent and add integrations column for encrypted credentials

-- First update all records to have empty JSON object
UPDATE "User"
SET    "metadata" = '{}'::jsonb
WHERE  "metadata" IS NULL;

-- Then make it required
ALTER TABLE "User"
ALTER COLUMN "metadata" SET DEFAULT '{}'::jsonb,
ALTER COLUMN "metadata" SET NOT NULL,
-- and add integrations column (which will be encrypted JSON)
ADD   COLUMN "integrations" TEXT NOT NULL DEFAULT '';

-- Encrypting the credentials and moving them from metadata to integrations
-- will be handled in the backend.
