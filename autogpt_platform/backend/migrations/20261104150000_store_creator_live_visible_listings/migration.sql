-- StoreCreator keeps dev's visible-listing rule and the branch's live-version rule.
--
-- 20261104120000_store_creator_counts_visible_listings rebuilt StoreCreator
-- from a definition without the branch's live-version rule (a listing whose
-- active version was deleted is not counted; 20260827170000, restored by
-- 20261101120000). On a fresh database that migration runs last and drops the
-- rule; on one where it already ran, 20261101120000 runs after it and puts back
-- the older view instead. Sorted after both, this rebuilds the view as
-- 20261104120000 defines it, plus that one condition. The columns and their
-- types are unchanged, so CREATE OR REPLACE keeps the view's dependants.

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
     AND active_slv."isDeleted" = false
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
