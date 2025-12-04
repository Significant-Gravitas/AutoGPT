-- Migration: Replace full-text search with pgvector-based vector search
-- This migration:
-- 1. Enables the pgvector extension
-- 2. Drops the StoreAgent view (depends on search column)
-- 3. Removes the full-text search infrastructure (trigger, function, tsvector column)
-- 4. Adds a vector embedding column for semantic search
-- 5. Creates an index for fast vector similarity search
-- 6. Recreates the StoreAgent view with the embedding column

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- First drop the view that depends on the search column
DROP VIEW IF EXISTS "StoreAgent";

-- Remove full-text search infrastructure
DROP TRIGGER IF EXISTS "update_tsvector" ON "StoreListingVersion";
DROP FUNCTION IF EXISTS update_tsvector_column();

-- Drop the tsvector search column
ALTER TABLE "StoreListingVersion" DROP COLUMN IF EXISTS "search";

-- Add embedding column for vector search (1536 dimensions for text-embedding-3-small)
ALTER TABLE "StoreListingVersion"
ADD COLUMN IF NOT EXISTS "embedding" vector(1536);

-- Create IVFFlat index for fast similarity search
-- Using cosine distance (vector_cosine_ops) which is standard for text embeddings
-- lists = 100 is appropriate for datasets under 1M rows
CREATE INDEX IF NOT EXISTS idx_store_listing_version_embedding
ON "StoreListingVersion"
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Recreate StoreAgent view WITHOUT search column, WITH embedding column
CREATE OR REPLACE VIEW "StoreAgent" AS
WITH latest_versions AS (
    SELECT
        "storeListingId",
        MAX(version) AS max_version
    FROM "StoreListingVersion"
    WHERE "submissionStatus" = 'APPROVED'
    GROUP BY "storeListingId"
),
agent_versions AS (
    SELECT
        "storeListingId",
        array_agg(DISTINCT version::text ORDER BY version::text) AS versions
    FROM "StoreListingVersion"
    WHERE "submissionStatus" = 'APPROVED'
    GROUP BY "storeListingId"
)
SELECT
    sl.id AS listing_id,
    slv.id AS "storeListingVersionId",
    slv."createdAt" AS updated_at,
    sl.slug,
    COALESCE(slv.name, '') AS agent_name,
    slv."videoUrl" AS agent_video,
    COALESCE(slv."imageUrls", ARRAY[]::text[]) AS agent_image,
    slv."isFeatured" AS featured,
    p.username AS creator_username,
    p."avatarUrl" AS creator_avatar,
    slv."subHeading" AS sub_heading,
    slv.description,
    slv.categories,
    slv.embedding,
    COALESCE(ar.run_count, 0::bigint) AS runs,
    COALESCE(rs.avg_rating, 0.0)::double precision AS rating,
    COALESCE(av.versions, ARRAY[slv.version::text]) AS versions,
    COALESCE(sl."useForOnboarding", false) AS "useForOnboarding",
    slv."isAvailable" AS is_available
FROM "StoreListing" sl
JOIN latest_versions lv
    ON sl.id = lv."storeListingId"
JOIN "StoreListingVersion" slv
    ON slv."storeListingId" = lv."storeListingId"
   AND slv.version = lv.max_version
   AND slv."submissionStatus" = 'APPROVED'
JOIN "AgentGraph" a
    ON slv."agentGraphId" = a.id
    AND slv."agentGraphVersion" = a.version
LEFT JOIN "Profile" p
    ON sl."owningUserId" = p."userId"
LEFT JOIN "mv_review_stats" rs
    ON sl.id = rs."storeListingId"
LEFT JOIN "mv_agent_run_counts" ar
    ON a.id = ar."agentGraphId"
LEFT JOIN agent_versions av
    ON sl.id = av."storeListingId"
WHERE sl."isDeleted" = false
  AND sl."hasApprovedVersion" = true;
