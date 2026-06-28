-- Update the StoreSubmission and StoreAgent views with additional fields, clearer field names, and faster joins.
-- Steps:
-- 1. Update `mv_agent_run_counts` to exclude runs by the agent's creator
--   a. Drop dependent views `StoreAgent` and `Creator`
--   b. Update `mv_agent_run_counts` and its index
--   c. Recreate `StoreAgent` view (with updates)
--   d. Restore `Creator` view
-- 2. Update `StoreSubmission` view
-- 3. Update `StoreListingReview` indices to make `StoreSubmission` query more efficient

BEGIN;

-- Drop views that are dependent on mv_agent_run_counts
DROP VIEW IF EXISTS "StoreAgent";
DROP VIEW IF EXISTS "Creator";

-- Update materialized view for agent run counts to exclude runs by the agent's creator
DROP INDEX IF EXISTS "idx_mv_agent_run_counts";
DROP MATERIALIZED VIEW IF EXISTS "mv_agent_run_counts";
CREATE MATERIALIZED VIEW "mv_agent_run_counts" AS
SELECT
    run."agentGraphId" AS graph_id,
    COUNT(*)           AS run_count
FROM "AgentGraphExecution" run
JOIN "AgentGraph"          graph ON graph.id = run."agentGraphId"
-- Exclude runs by the agent's creator to avoid inflating run counts
WHERE graph."userId" != run."userId"
GROUP BY run."agentGraphId";

-- Recreate index
CREATE UNIQUE INDEX IF NOT EXISTS "idx_mv_agent_run_counts" ON "mv_agent_run_counts"("graph_id");

-- Re-populate the materialized view
REFRESH MATERIALIZED VIEW "mv_agent_run_counts";


-- Recreate the StoreAgent view with the following changes
-- (compared to 20260115210000_remove_storelistingversion_search):
-- - Narrow to *explicitly active* version (sl.activeVersionId) instead of MAX(version)
-- - Add `recommended_schedule_cron` column
-- - Rename `"storeListingVersionId"` -> `listing_version_id`
-- - Rename `"agentGraphVersions"`    -> `graph_versions`
-- - Rename `"agentGraphId"`          -> `graph_id`
-- - Rename `"useForOnboarding"`      -> `use_for_onboarding`
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
     slv."recommendedScheduleCron"                       AS recommended_schedule_cron
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


-- Restore Creator view as last updated in 20250604130249_optimise_store_agent_and_creator_views,
-- with minor changes:
-- - Ensure top_categories always TEXT[]
-- - Filter out empty ('') categories
CREATE OR REPLACE VIEW "Creator" AS
WITH creator_listings AS (
    SELECT
        sl."owningUserId",
        sl.id AS listing_id,
        slv."agentGraphId",
        slv.categories,
        sr.score,
        ar.run_count
    FROM "StoreListing" sl
    JOIN "StoreListingVersion" slv
      ON slv."storeListingId" = sl.id
     AND slv."submissionStatus" = 'APPROVED'
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
    COALESCE(cs.agent_runs, 0::numeric)          AS agent_runs
FROM "Profile" p
LEFT JOIN creator_stats cs ON cs."owningUserId" = p."userId";


-- Recreate the StoreSubmission view with updated fields & query strategy:
-- - Uses mv_agent_run_counts instead of full AgentGraphExecution table scan + aggregation
-- - Renamed agent_id, agent_version -> graph_id, graph_version
-- - Renamed store_listing_version_id -> listing_version_id
-- - Renamed date_submitted -> submitted_at
-- - Renamed runs, rating -> run_count, review_avg_rating
-- - Added fields: instructions, agent_output_demo_url, review_count, is_deleted
DROP VIEW IF EXISTS "StoreSubmission";
CREATE OR REPLACE VIEW "StoreSubmission" AS
WITH review_stats AS (
    SELECT
        "storeListingVersionId" AS version_id, -- more specific than mv_review_stats
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
    COALESCE(review_stats.avg_rating, 0.0)::double precision AS review_avg_rating
FROM      "StoreListing"        sl
JOIN      "StoreListingVersion" slv     ON slv."storeListingId" = sl.id
LEFT JOIN review_stats                  ON review_stats.version_id = slv.id
LEFT JOIN mv_agent_run_counts run_stats ON run_stats.graph_id = slv."agentGraphId"
WHERE     sl."isDeleted" = false;


-- Drop unused index on StoreListingReview.reviewByUserId
DROP INDEX IF EXISTS "StoreListingReview_reviewByUserId_idx";
-- Add index on storeListingVersionId to make StoreSubmission query faster
CREATE INDEX "StoreListingReview_storeListingVersionId_idx" ON "StoreListingReview"("storeListingVersionId");

COMMIT;
