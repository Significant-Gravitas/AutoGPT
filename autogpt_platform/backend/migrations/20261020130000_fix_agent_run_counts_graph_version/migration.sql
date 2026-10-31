-- Fix mv_agent_run_counts multiplying runs by the number of graph versions
--
-- The previous definition (20260304123456_update_store_views) joined
-- "AgentGraph" on id only. AgentGraph's key is (id, version), so every
-- execution matched one row per saved version of its graph and run_count
-- came out as true runs x graph versions. That inflated StoreAgent.runs,
-- StoreSubmission.run_count and StoreCreator.agent_runs.
--
-- Join on the full key instead. Every execution has a NOT NULL foreign key on
-- ("agentGraphId", "agentGraphVersion") to AgentGraph, so each run matches
-- exactly one row and the owner filter still compares against the owner of
-- the version that ran.
--
-- A materialized view's query cannot be replaced in place, so the dependent
-- views are dropped and recreated: StoreAgent, Creator and StoreSubmission
-- unchanged from 20260415000000_org_tenancy_cutover, StoreCreator as defined
-- by 20261020120000_fix_store_creator_stats (#15762). refresh_store_materialized_views() and
-- its pg_cron job reference the view by name and need no change.

-- Drop views that depend on mv_agent_run_counts (and Creator, which reads StoreCreator)
DROP VIEW IF EXISTS "Creator";
DROP VIEW IF EXISTS "StoreCreator";
DROP VIEW IF EXISTS "StoreAgent";
DROP VIEW IF EXISTS "StoreSubmission";

-- Rebuild the run counts, joining AgentGraph on (id, version)
DROP INDEX IF EXISTS "idx_mv_agent_run_counts";
DROP MATERIALIZED VIEW IF EXISTS "mv_agent_run_counts";
CREATE MATERIALIZED VIEW "mv_agent_run_counts" AS
SELECT
    run."agentGraphId" AS graph_id,
    COUNT(*)           AS run_count
FROM "AgentGraphExecution" run
JOIN "AgentGraph"          graph ON graph.id      = run."agentGraphId"
                                AND graph.version = run."agentGraphVersion"
-- Exclude runs by the agent's creator to avoid inflating run counts
WHERE graph."userId" != run."userId"
GROUP BY run."agentGraphId";

CREATE UNIQUE INDEX IF NOT EXISTS "idx_mv_agent_run_counts" ON "mv_agent_run_counts"("graph_id");

-- Recreate the dependent views (StoreCreator from 20261020120000_fix_store_creator_stats)

-- 4b. Recreate StoreAgent view with owning_org_id
CREATE OR REPLACE VIEW "StoreAgent" AS
WITH store_agent_versions AS (
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
     sl.id                                               AS listing_id,
     slv.id                                              AS listing_version_id,
     slv."createdAt"                                     AS updated_at,
     sl.slug,
     COALESCE(slv.name, '')                              AS agent_name,
     slv."videoUrl"                                      AS agent_video,
     slv."agentOutputDemoUrl"                            AS agent_output_demo,
     COALESCE(slv."imageUrls", ARRAY[]::text[])          AS agent_image,
     slv."isFeatured"                                    AS featured,
     cp.username                                         AS creator_username,
     cp."avatarUrl"                                      AS creator_avatar,
     slv."subHeading"                                    AS sub_heading,
     slv.description,
     slv.categories,
     COALESCE(arc.run_count, 0::bigint)                  AS runs,
     COALESCE(reviews.avg_rating, 0.0)::double precision AS rating,
     COALESCE(sav.versions, ARRAY[slv.version::text])    AS versions,
     slv."agentGraphId"                                  AS graph_id,
     COALESCE(
        agv.graph_versions,
        ARRAY[slv."agentGraphVersion"::text]
     )                                                   AS graph_versions,
     slv."isAvailable"                                   AS is_available,
     COALESCE(sl."useForOnboarding", false)              AS use_for_onboarding,
     slv."recommendedScheduleCron"                       AS recommended_schedule_cron,
     sl."owningOrgId"                                    AS owning_org_id
FROM "StoreListing"             AS sl
JOIN "StoreListingVersion"      AS slv
  ON slv."storeListingId" = sl.id
 AND slv.id = sl."activeVersionId"
 AND slv."submissionStatus" = 'APPROVED'
JOIN "AgentGraph"               AS ag
  ON slv."agentGraphId" = ag.id
 AND slv."agentGraphVersion" = ag.version
LEFT JOIN "Profile"             AS cp
  ON sl."owningUserId" = cp."userId"
LEFT JOIN "mv_review_stats"     AS reviews
  ON sl.id = reviews."storeListingId"
LEFT JOIN "mv_agent_run_counts" AS arc
  ON ag.id = arc.graph_id
LEFT JOIN store_agent_versions  AS sav
  ON sl.id = sav."storeListingId"
LEFT JOIN agent_graph_versions  AS agv
  ON sl.id = agv."storeListingId"
WHERE sl."isDeleted" = false
  AND sl."hasApprovedVersion" = true;


-- 4c. StoreCreator, carried verbatim from 20261020120000_fix_store_creator_stats
--     (#15762) so this migration cannot revert that fix in either merge order.
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


-- 4d. Keep original Creator view for backward compatibility
CREATE OR REPLACE VIEW "Creator" AS
SELECT
    username,
    name,
    avatar_url,
    description,
    top_categories,
    links,
    is_featured,
    num_agents,
    agent_rating,
    agent_runs
FROM "StoreCreator";


-- 4e. Recreate StoreSubmission view with organization_id
CREATE OR REPLACE VIEW "StoreSubmission" AS
WITH review_stats AS (
    SELECT
        "storeListingVersionId" AS version_id,
        avg(score)              AS avg_rating,
        count(*)                AS review_count
    FROM "StoreListingReview"
    GROUP BY "storeListingVersionId"
)
SELECT
    sl.id              AS listing_id,
    sl."owningUserId"  AS user_id,
    sl.slug            AS slug,

    slv.id                   AS listing_version_id,
    slv.version              AS listing_version,
    slv."agentGraphId"       AS graph_id,
    slv."agentGraphVersion"  AS graph_version,
    slv.name                 AS name,
    slv."subHeading"         AS sub_heading,
    slv.description          AS description,
    slv.instructions         AS instructions,
    slv.categories           AS categories,
    slv."imageUrls"          AS image_urls,
    slv."videoUrl"           AS video_url,
    slv."agentOutputDemoUrl" AS agent_output_demo_url,
    slv."submittedAt"        AS submitted_at,
    slv."changesSummary"     AS changes_summary,
    slv."submissionStatus"   AS status,
    slv."reviewedAt"         AS reviewed_at,
    slv."reviewerId"         AS reviewer_id,
    slv."reviewComments"     AS review_comments,
    slv."internalComments"   AS internal_comments,
    slv."isDeleted"          AS is_deleted,

    COALESCE(run_stats.run_count, 0::bigint) AS run_count,
    COALESCE(review_stats.review_count, 0::bigint) AS review_count,
    COALESCE(review_stats.avg_rating, 0.0)::double precision AS review_avg_rating,

    sl."owningOrgId"         AS organization_id
FROM      "StoreListing"        sl
JOIN      "StoreListingVersion" slv     ON slv."storeListingId" = sl.id
LEFT JOIN review_stats                  ON review_stats.version_id = slv.id
LEFT JOIN mv_agent_run_counts run_stats ON run_stats.graph_id = slv."agentGraphId"
WHERE     sl."isDeleted" = false;
