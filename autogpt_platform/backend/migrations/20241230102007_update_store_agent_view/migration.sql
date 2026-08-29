BEGIN;

DROP VIEW IF EXISTS "StoreAgent";

CREATE VIEW "StoreAgent" AS
WITH ReviewStats AS (
    SELECT sl."id" AS "storeListingId",
    COUNT(sr.id) AS review_count,
    AVG(CAST(sr.score AS DECIMAL)) AS avg_rating
    FROM "StoreListing" sl
    JOIN "StoreListingVersion" slv ON slv."storeListingId" = sl."id"
    JOIN "StoreListingReview" sr ON sr."storeListingVersionId" = slv.id
    WHERE sl."isDeleted" = FALSE
    GROUP BY sl."id"
),
AgentRuns AS (
    SELECT "agentGraphId", COUNT(*) AS run_count
    FROM "AgentGraphExecution"
    GROUP BY "agentGraphId"
)
SELECT
    sl.id AS listing_id,
    slv.id AS "storeListingVersionId",
    slv."createdAt" AS updated_at,
    slv.slug,
    slv.name AS agent_name,
    slv."videoUrl" AS agent_video,
    COALESCE(slv."imageUrls", ARRAY[]::TEXT[]) AS agent_image,
    slv."isFeatured" AS featured,
    p.username AS creator_username,
    p."avatarUrl" AS creator_avatar,
    slv."subHeading" AS sub_heading,
    slv.description,
    slv.categories,
    COALESCE(ar.run_count, 0) AS runs,
    CAST(COALESCE(rs.avg_rating, 0.0) AS DOUBLE PRECISION) AS rating,
    ARRAY_AGG(DISTINCT CAST(slv.version AS TEXT)) AS versions
FROM "StoreListing" sl
JOIN "AgentGraph" a ON sl."agentId" = a.id AND sl."agentVersion" = a."version"
LEFT JOIN "Profile" p ON sl."owningUserId" = p."userId"
LEFT JOIN "StoreListingVersion" slv ON slv."storeListingId" = sl.id
LEFT JOIN ReviewStats rs ON sl.id = rs."storeListingId"
LEFT JOIN AgentRuns ar ON a.id = ar."agentGraphId"
WHERE sl."isDeleted" = FALSE
  AND sl."isApproved" = TRUE
GROUP BY sl.id, slv.id, slv.slug, slv."createdAt", slv.name, slv."videoUrl", slv."imageUrls", slv."isFeatured", 
         p.username, p."avatarUrl", slv."subHeading", slv.description, slv.categories,
         ar.run_count, rs.avg_rating;

COMMIT;
