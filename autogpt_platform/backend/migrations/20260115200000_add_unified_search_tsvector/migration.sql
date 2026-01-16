-- Add tsvector search column to UnifiedContentEmbedding for unified full-text search
-- This enables hybrid search (semantic + lexical) across all content types

-- Add search column (IF NOT EXISTS for idempotency)
ALTER TABLE "UnifiedContentEmbedding" ADD COLUMN IF NOT EXISTS "search" tsvector DEFAULT ''::tsvector;

-- NOTE: No GIN index created here - same approach as StoreListingVersion.search
-- The tsvector column works without an index (sequential scan)
-- Performance comes from the HNSW embedding index for semantic search
-- If needed later, GIN index can be added outside of Prisma migrations

-- Drop existing trigger/function if exists
DROP TRIGGER IF EXISTS "update_unified_tsvector" ON "UnifiedContentEmbedding";
DROP FUNCTION IF EXISTS update_unified_tsvector_column();

-- Create function to auto-update tsvector from searchableText
CREATE OR REPLACE FUNCTION update_unified_tsvector_column() RETURNS TRIGGER AS $$
BEGIN
  NEW.search := to_tsvector('english', COALESCE(NEW."searchableText", ''));
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path = platform, pg_temp;

-- Create trigger to auto-update search column on insert/update
CREATE TRIGGER "update_unified_tsvector"
BEFORE INSERT OR UPDATE ON "UnifiedContentEmbedding"
FOR EACH ROW
EXECUTE FUNCTION update_unified_tsvector_column();

-- Backfill existing rows
UPDATE "UnifiedContentEmbedding"
SET search = to_tsvector('english', COALESCE("searchableText", ''))
WHERE search IS NULL OR search = ''::tsvector;
