-- Remove the old search column from StoreListingVersion
-- This column has been replaced by UnifiedContentEmbedding.search
-- which provides unified hybrid search across all content types

-- First drop the dependent view
DROP VIEW IF EXISTS "StoreAgent";

-- Drop the trigger and function for old search column
-- The original trigger was created in 20251016093049_add_full_text_search
DROP TRIGGER IF EXISTS "update_tsvector" ON "StoreListingVersion";
DROP FUNCTION IF EXISTS update_tsvector_column();

-- Drop the index
DROP INDEX IF EXISTS "StoreListingVersion_search_idx";

-- NOTE: Keeping search column for now to allow easy revert if needed
-- Uncomment to fully remove once migration is verified in production:
-- ALTER TABLE "StoreListingVersion" DROP COLUMN IF EXISTS "search";

-- Recreate the StoreAgent view WITHOUT the search column
-- (Search now handled by UnifiedContentEmbedding)
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
),
agent_graph_versions AS (
    SELECT
        "storeListingId",
        array_agg(DISTINCT "agentGraphVersion"::text ORDER BY "agentGraphVersion"::text) AS graph_versions
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
    slv."agentOutputDemoUrl" AS agent_output_demo,
    COALESCE(slv."imageUrls", ARRAY[]::text[]) AS agent_image,
    slv."isFeatured" AS featured,
    p.username AS creator_username,
    p."avatarUrl" AS creator_avatar,
    slv."subHeading" AS sub_heading,
    slv.description,
    slv.categories,
    COALESCE(ar.run_count, 0::bigint) AS runs,
    COALESCE(rs.avg_rating, 0.0)::double precision AS rating,
    COALESCE(av.versions, ARRAY[slv.version::text]) AS versions,
    COALESCE(agv.graph_versions, ARRAY[slv."agentGraphVersion"::text]) AS "agentGraphVersions",
    slv."agentGraphId",
    slv."isAvailable" AS is_available,
    COALESCE(sl."useForOnboarding", false) AS "useForOnboarding"
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
LEFT JOIN agent_graph_versions agv
    ON sl.id = agv."storeListingId"
WHERE sl."isDeleted" = false
  AND sl."hasApprovedVersion" = true;
