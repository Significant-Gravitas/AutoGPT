CREATE OR REPLACE VIEW "StoreCreator" AS
WITH creator_listings AS (
    SELECT
        sl."owningUserId",
        sl."owningOrgId",
        sl.id AS listing_id,
        slv."agentGraphId",
        slv.categories,
        sr.score,
        ar.run_count
    FROM "StoreListing" sl
    JOIN "StoreListingVersion" slv
      ON slv.id = sl."activeVersionId"
     AND slv."storeListingId" = sl.id
     AND slv."submissionStatus" = 'APPROVED'
     AND slv."isAvailable" = true
     AND slv."isDeleted" = false
    LEFT JOIN "StoreListingReview" sr
           ON sr."storeListingVersionId" = slv.id
    LEFT JOIN "mv_agent_run_counts" ar
           ON ar.graph_id = slv."agentGraphId"
    WHERE sl."isDeleted" = false
      AND sl."hasApprovedVersion" = true
),
creator_stats AS (
    SELECT
        cl."owningUserId",
        COUNT(DISTINCT cl.listing_id)                  AS num_agents,
        AVG(COALESCE(cl.score, 0)::numeric)            AS agent_rating,
        SUM(COALESCE(cl.run_count, 0))                 AS agent_runs,
        array_agg(DISTINCT cat ORDER BY cat)
          FILTER (WHERE cat IS NOT NULL AND cat != '') AS all_categories
    FROM creator_listings cl
    LEFT JOIN LATERAL unnest(COALESCE(cl.categories, ARRAY[]::text[])) AS cat ON true
    GROUP BY cl."owningUserId"
)
SELECT
    p.username,
    p.name,
    p."avatarUrl"                                AS avatar_url,
    p.description,
    COALESCE(cs.all_categories, ARRAY[]::text[]) AS top_categories,
    p.links,
    p."isFeatured"                               AS is_featured,
    COALESCE(cs.num_agents, 0::bigint)           AS num_agents,
    COALESCE(cs.agent_rating, 0.0)               AS agent_rating,
    COALESCE(cs.agent_runs, 0::numeric)          AS agent_runs,
    o.name                                       AS org_name
FROM "Profile" p
LEFT JOIN creator_stats cs ON cs."owningUserId" = p."userId"
LEFT JOIN "OrgMember" om ON om."userId" = p."userId" AND om."isOwner" = true
LEFT JOIN "Organization" o ON o.id = om."orgId" AND o."isPersonal" = true;
