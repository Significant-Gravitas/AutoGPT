-- Remove the old search column from StoreListingVersion
-- This column has been replaced by UnifiedContentEmbedding.search
-- which provides unified hybrid search across all content types

-- Drop the trigger first
DROP TRIGGER IF EXISTS "update_store_listing_version_tsvector" ON "StoreListingVersion";

-- Drop the function
DROP FUNCTION IF EXISTS update_store_listing_version_tsvector_column();

-- Drop the index
DROP INDEX IF EXISTS "StoreListingVersion_search_idx";

-- Remove the search column
ALTER TABLE "StoreListingVersion" DROP COLUMN IF EXISTS "search";
