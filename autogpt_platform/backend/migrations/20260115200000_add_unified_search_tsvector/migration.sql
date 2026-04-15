-- Add tsvector search column to UnifiedContentEmbedding for unified full-text search
-- This enables hybrid search (semantic + lexical) across all content types

-- Add search column (IF NOT EXISTS for idempotency)
ALTER TABLE "UnifiedContentEmbedding" ADD COLUMN IF NOT EXISTS "search" tsvector DEFAULT ''::tsvector;

-- Create GIN index for fast full-text search
-- No @@index in schema.prisma - Prisma may generate DROP INDEX on migrate dev
-- If that happens, just let it drop and this migration will recreate it, or manually re-run:
--   CREATE INDEX IF NOT EXISTS "UnifiedContentEmbedding_search_idx" ON "UnifiedContentEmbedding" USING GIN ("search");
DROP INDEX IF EXISTS "UnifiedContentEmbedding_search_idx";
CREATE INDEX "UnifiedContentEmbedding_search_idx" ON "UnifiedContentEmbedding" USING GIN ("search");

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
