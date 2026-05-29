BEGIN;
-- AlterTable
ALTER TABLE "StoreListing" ADD COLUMN     "useForOnboarding" BOOLEAN NOT NULL DEFAULT false;

-- Drop and recreate the StoreAgent view with useForOnboarding field
DROP VIEW IF EXISTS "StoreAgent";

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
    p.username AS creator_username,  -- Allow NULL for malformed sub-agents
    p."avatarUrl" AS creator_avatar,  -- Allow NULL for malformed sub-agents
    slv."subHeading" AS sub_heading,
    slv.description,
    slv.categories,
    COALESCE(ar.run_count, 0::bigint) AS runs,
    COALESCE(rs.avg_rating, 0.0)::double precision AS rating,
    COALESCE(av.versions, ARRAY[slv.version::text]) AS versions,
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
WHERE sl."isDeleted" = false
  AND sl."hasApprovedVersion" = true;

COMMIT;