-- DropView
DROP VIEW IF EXISTS "StoreAgent";

-- CreateView
CREATE OR REPLACE VIEW "StoreAgent" AS
WITH latest_versions AS (
    SELECT
        "storeListingId",
        MAX(version) AS latest_version
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
    COALESCE(ar.run_count, 0::bigint) AS runs,
    COALESCE(rs.avg_rating, 0.0)::double precision AS rating,
    ARRAY[slv.version::text] AS versions
FROM "StoreListing" sl
INNER JOIN latest_versions lv
    ON lv."storeListingId" = sl.id
INNER JOIN "StoreListingVersion" slv
    ON slv."storeListingId" = sl.id
    AND slv.version = lv.latest_version
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
WHERE sl."isDeleted" = false
  AND sl."hasApprovedVersion" = true;
