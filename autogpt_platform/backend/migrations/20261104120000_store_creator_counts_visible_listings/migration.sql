-- StoreCreator counts only the listings the marketplace shows (SECRT-2057)
--
-- creator_listings took every live listing with any approved version, so a
-- creator's num_agents counted listings whose active version was hidden
-- (isAvailable = false) or not approved. The creator page lists agents from
-- StoreAgent filtered on is_available, which drops those, so the card said
-- more workflows than the page showed.
--
-- creator_listings now selects the same set as StoreAgent with
-- is_available = true: the active version is approved and available, and its
-- graph exists. Runs, rating and categories read from that set too, the way
-- StoreAgent reads them: runs by the active version's graph, rating over the
-- listing's approved versions (as mv_review_stats does), categories from the
-- active version. So every number on the card describes the listings the
-- page shows. The columns and their types are unchanged, so CREATE OR REPLACE
-- keeps the Creator view and the Prisma view models as they are.

CREATE OR REPLACE VIEW "StoreCreator" AS
WITH creator_listings AS (
    SELECT
        sl.id                    AS listing_id,
        sl."owningUserId",
        active_slv.id            AS active_version_id,
        active_slv."agentGraphId"
    FROM "StoreListing" sl
    JOIN "StoreListingVersion" active_slv
      ON active_slv."storeListingId" = sl.id
     AND active_slv.id = sl."activeVersionId"
     AND active_slv."submissionStatus" = 'APPROVED'
     AND active_slv."isAvailable" = true
    JOIN "AgentGraph" ag
      ON ag.id = active_slv."agentGraphId"
     AND ag.version = active_slv."agentGraphVersion"
    WHERE sl."isDeleted" = false
      AND sl."hasApprovedVersion" = true
),
creator_runs AS (
    SELECT
        cl."owningUserId",
        COUNT(*)                        AS num_agents,
        SUM(COALESCE(ar.run_count, 0))  AS agent_runs
    FROM creator_listings cl
    LEFT JOIN "mv_agent_run_counts" ar
           ON ar.graph_id = cl."agentGraphId"
    GROUP BY cl."owningUserId"
),
creator_ratings AS (
    SELECT
        cl."owningUserId",
        AVG(sr.score::numeric) AS agent_rating
    FROM creator_listings cl
    JOIN "StoreListingVersion" slv
      ON slv."storeListingId" = cl.listing_id
     AND slv."submissionStatus" = 'APPROVED'
    JOIN "StoreListingReview" sr
      ON sr."storeListingVersionId" = slv.id
    GROUP BY cl."owningUserId"
),
creator_categories AS (
    SELECT
        cl."owningUserId",
        array_agg(DISTINCT cat ORDER BY cat)
          FILTER (WHERE cat IS NOT NULL AND cat != '') AS all_categories
    FROM creator_listings cl
    JOIN "StoreListingVersion" slv
      ON slv.id = cl.active_version_id
    LEFT JOIN LATERAL unnest(COALESCE(slv.categories, ARRAY[]::text[])) AS cat ON true
    GROUP BY cl."owningUserId"
)
SELECT
    p.username,
    p.name,
    p."avatarUrl"                                AS avatar_url,
    p.description,
    COALESCE(cc.all_categories, ARRAY[]::text[]) AS top_categories,
    p.links,
    p."isFeatured"                               AS is_featured,
    COALESCE(cr.num_agents, 0::bigint)           AS num_agents,
    COALESCE(crt.agent_rating, 0.0)              AS agent_rating,
    COALESCE(cr.agent_runs, 0::numeric)          AS agent_runs,
    o.name                                       AS org_name
FROM "Profile" p
LEFT JOIN creator_runs cr        ON cr."owningUserId" = p."userId"
LEFT JOIN creator_ratings crt    ON crt."owningUserId" = p."userId"
LEFT JOIN creator_categories cc  ON cc."owningUserId" = p."userId"
LEFT JOIN "OrgMember" om ON om."userId" = p."userId" AND om."isOwner" = true
LEFT JOIN "Organization" o ON o.id = om."orgId" AND o."isPersonal" = true;
