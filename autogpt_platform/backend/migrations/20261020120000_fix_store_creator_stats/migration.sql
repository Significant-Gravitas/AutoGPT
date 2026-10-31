-- Fix StoreCreator stats fan-out (#15760)
--
-- The previous definition joined every APPROVED version, every review on
-- those versions and the per-graph run count into one row set, then
-- unnested categories on top. That made one row per
-- listing x approved version x review x category, so:
--   * agent_runs summed the same per-graph run count once per row;
--   * agent_rating averaged a 0 in for each unreviewed approved version and
--     repeated each review once per category.
--
-- Each stat now has its own aggregate:
--   * num_agents: live listings with an approved version (unchanged);
--   * agent_runs: mv_agent_run_counts once per listing, via
--     StoreListing.agentGraphId (unique per listing);
--   * agent_rating: review-weighted average of real review scores on the
--     creator's approved versions; NULL inside, 0.0 only in the final select;
--   * top_categories: distinct, sorted, non-empty categories across approved
--     versions (unchanged).
--
-- Column names, order and types are identical, so CREATE OR REPLACE keeps the
-- dependent "Creator" view and both Prisma views as they are. No materialized
-- view changes, so no refresh is needed.

CREATE OR REPLACE VIEW "StoreCreator" AS
WITH creator_listings AS (
    SELECT
        sl.id AS listing_id,
        sl."owningUserId",
        sl."agentGraphId"
    FROM "StoreListing" sl
    WHERE sl."isDeleted" = false
      AND sl."hasApprovedVersion" = true
      AND EXISTS (
          SELECT 1
          FROM "StoreListingVersion" slv
          WHERE slv."storeListingId" = sl.id
            AND slv."submissionStatus" = 'APPROVED'
      )
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
      ON slv."storeListingId" = cl.listing_id
     AND slv."submissionStatus" = 'APPROVED'
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
