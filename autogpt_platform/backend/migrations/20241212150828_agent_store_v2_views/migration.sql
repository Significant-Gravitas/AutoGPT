BEGIN;

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
    a.name AS agent_name,
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
GROUP BY sl.id, slv.id, slv.slug, slv."createdAt", a.name, slv."videoUrl", slv."imageUrls", slv."isFeatured", 
         p.username, p."avatarUrl", slv."subHeading", slv.description, slv.categories,
         ar.run_count, rs.avg_rating;

CREATE VIEW "Creator" AS
WITH AgentStats AS (
    SELECT 
        p.username,
        COUNT(DISTINCT sl.id) as num_agents,
        AVG(CAST(COALESCE(sr.score, 0) AS DECIMAL)) as agent_rating,
        SUM(COALESCE(age.run_count, 0)) as agent_runs
    FROM "Profile" p
    LEFT JOIN "StoreListing" sl ON sl."owningUserId" = p."userId"
    LEFT JOIN "StoreListingVersion" slv ON slv."storeListingId" = sl.id
    LEFT JOIN "StoreListingReview" sr ON sr."storeListingVersionId" = slv.id
    LEFT JOIN (
        SELECT "agentGraphId", COUNT(*) as run_count 
        FROM "AgentGraphExecution"
        GROUP BY "agentGraphId"
    ) age ON age."agentGraphId" = sl."agentId"
    WHERE sl."isDeleted" = FALSE AND sl."isApproved" = TRUE
    GROUP BY p.username
)
SELECT
    p.username,
    p.name,
    p."avatarUrl" as avatar_url,
    p.description,
    ARRAY_AGG(DISTINCT c) FILTER (WHERE c IS NOT NULL) as top_categories,
    p.links,
    p."isFeatured" as is_featured,
    COALESCE(ast.num_agents, 0) as num_agents,
    COALESCE(ast.agent_rating, 0.0) as agent_rating,
    COALESCE(ast.agent_runs, 0) as agent_runs
FROM "Profile" p
LEFT JOIN AgentStats ast ON ast.username = p.username
LEFT JOIN LATERAL (
    SELECT UNNEST(slv.categories) as c
    FROM "StoreListing" sl
    JOIN "StoreListingVersion" slv ON slv."storeListingId" = sl.id
    WHERE sl."owningUserId" = p."userId"
    AND sl."isDeleted" = FALSE 
    AND sl."isApproved" = TRUE
) cats ON true
GROUP BY p.username, p.name, p."avatarUrl", p.description, p.links, p."isFeatured",
         ast.num_agents, ast.agent_rating, ast.agent_runs;

CREATE VIEW "StoreSubmission" AS
SELECT
    sl.id as listing_id,
    sl."owningUserId" as user_id,
    slv."agentId" as agent_id,
    slv."version" as agent_version,
    slv.slug,
    slv.name,
    slv."subHeading" as sub_heading,
    slv.description,
    slv."imageUrls" as image_urls,
    slv."createdAt" as date_submitted,
    COALESCE(sls."Status", 'PENDING') as status,
    COALESCE(ar.run_count, 0) as runs,
    CAST(COALESCE(AVG(CAST(sr.score AS DECIMAL)), 0.0) AS DOUBLE PRECISION) as rating
FROM "StoreListing" sl
JOIN "StoreListingVersion" slv ON slv."storeListingId" = sl.id
LEFT JOIN "StoreListingSubmission" sls ON sls."storeListingId" = sl.id
LEFT JOIN "StoreListingReview" sr ON sr."storeListingVersionId" = slv.id
LEFT JOIN (
    SELECT "agentGraphId", COUNT(*) as run_count
    FROM "AgentGraphExecution" 
    GROUP BY "agentGraphId"
) ar ON ar."agentGraphId" = slv."agentId"
WHERE sl."isDeleted" = FALSE
GROUP BY sl.id, sl."owningUserId", slv."agentId", slv."version", slv.slug, slv.name, slv."subHeading", 
         slv.description, slv."imageUrls", slv."createdAt", sls."Status", ar.run_count;

COMMIT;