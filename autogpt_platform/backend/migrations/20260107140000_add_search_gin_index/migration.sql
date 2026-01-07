-- Add GIN index on StoreListingVersion.search for fast full-text search
-- This index is critical for hybrid search performance

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_store_listing_version_search_gin
ON "StoreListingVersion" USING GIN (search)
WHERE "submissionStatus" = 'APPROVED';
