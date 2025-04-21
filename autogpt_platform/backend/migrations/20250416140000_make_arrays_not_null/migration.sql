-- Backfill nulls with empty arrays
UPDATE "UserOnboarding"
SET "integrations" = ARRAY[]::TEXT[]
WHERE "integrations" IS NULL;

UPDATE "UserOnboarding"
SET "completedSteps" = '{}' 
WHERE "completedSteps" IS NULL;

UPDATE "UserOnboarding"
SET "notified" = '{}' 
WHERE "notified" IS NULL;

UPDATE "UserOnboarding"
SET "rewardedFor" = '{}' 
WHERE "rewardedFor" IS NULL;

UPDATE "IntegrationWebhook"
SET "events" = ARRAY[]::TEXT[]
WHERE "events" IS NULL;

UPDATE "APIKey"
SET "permissions" = '{}' 
WHERE "permissions" IS NULL;

UPDATE "Profile"
SET "links" = ARRAY[]::TEXT[]
WHERE "links" IS NULL;

UPDATE "StoreListingVersion"
SET "imageUrls" = ARRAY[]::TEXT[]
WHERE "imageUrls" IS NULL;

UPDATE "StoreListingVersion"
SET "categories" = ARRAY[]::TEXT[]
WHERE "categories" IS NULL;

-- Enforce NOT NULL constraints
ALTER TABLE "UserOnboarding"
    ALTER COLUMN "integrations" SET NOT NULL,
    ALTER COLUMN "completedSteps" SET NOT NULL,
    ALTER COLUMN "notified" SET NOT NULL,
    ALTER COLUMN "rewardedFor" SET NOT NULL;

ALTER TABLE "IntegrationWebhook"
    ALTER COLUMN "events" SET NOT NULL;

ALTER TABLE "APIKey"
    ALTER COLUMN "permissions" SET NOT NULL;

ALTER TABLE "Profile"
    ALTER COLUMN "links" SET NOT NULL;

ALTER TABLE "StoreListingVersion"
    ALTER COLUMN "imageUrls" SET NOT NULL,
    ALTER COLUMN "categories" SET NOT NULL;
