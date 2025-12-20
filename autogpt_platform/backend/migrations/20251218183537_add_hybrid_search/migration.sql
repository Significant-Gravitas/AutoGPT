-- Migration: Add hybrid search infrastructure (BM25 + vector + popularity)
-- This migration:
-- 1. Creates/updates the tsvector trigger with weighted fields
-- 2. Adds GIN index for full-text search performance
-- 3. Backfills existing records with tsvector data

-- Create or replace the trigger function with WEIGHTED tsvector
-- Weight A = name (highest priority), B = subHeading, C = description
CREATE OR REPLACE FUNCTION update_tsvector_column() RETURNS TRIGGER AS $$
BEGIN
  NEW.search := setweight(to_tsvector('english', COALESCE(NEW.name, '')), 'A') ||
                setweight(to_tsvector('english', COALESCE(NEW."subHeading", '')), 'B') ||
                setweight(to_tsvector('english', COALESCE(NEW.description, '')), 'C');
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop and recreate trigger to ensure it's active with the updated function
DROP TRIGGER IF EXISTS "update_tsvector" ON "StoreListingVersion";
CREATE TRIGGER "update_tsvector"
BEFORE INSERT OR UPDATE OF name, "subHeading", description ON "StoreListingVersion"
FOR EACH ROW
EXECUTE FUNCTION update_tsvector_column();

-- Create GIN index for full-text search performance
CREATE INDEX IF NOT EXISTS idx_store_listing_version_search_gin
ON "StoreListingVersion" USING GIN (search);

-- Backfill existing records with weighted tsvector
UPDATE "StoreListingVersion"
SET search = setweight(to_tsvector('english', COALESCE(name, '')), 'A') ||
             setweight(to_tsvector('english', COALESCE("subHeading", '')), 'B') ||
             setweight(to_tsvector('english', COALESCE(description, '')), 'C')
WHERE search IS NULL
   OR search = ''::tsvector;
