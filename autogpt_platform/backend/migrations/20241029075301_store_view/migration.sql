BEGIN;

CREATE VIEW "StoreAgent" AS
WITH ReviewStats AS (
    SELECT 
        sl."id" AS "storeListingId",
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
    sl."updatedAt" AS "lastUpdated",
    slv.version AS "version",
    
    sl."agentId" AS "agentId",
    a.name AS "agentName",
    a."version" AS "agentVersion",
    
    p.username AS "creatorName",
    p."avatarUrl" AS "avatarSrc",
    
    slv."isFeatured",
    slv.slug,
    slv.name,
    slv.description,
    
    slv."videoUrl" AS "videoUrl",
    COALESCE(slv."imageUrls", ARRAY[]::TEXT[]) AS "imageUrls",

    slv.categories,
    COALESCE(ar.run_count, 0) AS runs,
    CAST(COALESCE(rs.avg_rating, 0.0) AS DOUBLE PRECISION) AS rating
FROM "StoreListing" sl
JOIN "AgentGraph" a ON sl."agentId" = a.id AND sl."agentVersion" = a."version"
LEFT JOIN "Profile" p ON sl."owningUserId" = p."userId"
LEFT JOIN LATERAL (
    SELECT slv.*
    FROM "StoreListingVersion" slv
    WHERE slv."storeListingId" = sl.id
    ORDER BY slv."updatedAt" DESC
    LIMIT 1
) slv ON TRUE
LEFT JOIN ReviewStats rs ON sl.id = rs."storeListingId"
LEFT JOIN AgentRuns ar ON a.id = ar."agentGraphId"
WHERE sl."isDeleted" = FALSE
  AND sl."isApproved" = TRUE;

COMMIT;