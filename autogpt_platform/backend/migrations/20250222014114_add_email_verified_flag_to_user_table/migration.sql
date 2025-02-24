-- First, add the column as nullable to avoid issues with existing rows
ALTER TABLE "User" ADD COLUMN "emailVerified" BOOLEAN;

-- Set default values for existing rows
UPDATE "User" SET "emailVerified" = true;

-- Now make it NOT NULL and set the default
ALTER TABLE "User" ALTER COLUMN "emailVerified" SET NOT NULL;
ALTER TABLE "User" ALTER COLUMN "emailVerified" SET DEFAULT true;

