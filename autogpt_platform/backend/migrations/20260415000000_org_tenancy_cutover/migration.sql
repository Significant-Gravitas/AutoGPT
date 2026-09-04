-- Org tenancy cutover migration
-- 1. (REMOVED) Backfill of NULL organizationId — owned by the app-startup
--    backfill (backend/data/org_migration.py), NOT this migration
-- 2. Make organizationId NOT NULL on those tables (deferred)
-- 3. Update unique constraint on StoreListing from (owningUserId, slug) to (owningOrgId, slug) (deferred)
-- 4. Update views: StoreAgent (add owning_org_id), StoreSubmission (add organization_id),
--    Creator → StoreCreator (add org_name)

-- Prisma wraps each migration file in its own transaction.

-- ==========================================================================
-- Step 1: (REMOVED) Backfill NULL organizationId values
-- ==========================================================================
-- The backfill CANNOT do useful work at migration time: personal orgs are
-- created by the app-startup backfill (org_migration.py), which runs AFTER
-- migrations — so Organization/OrgMember are always empty here and the
-- per-row subquery resolved to NULL for every row. On the dev database that
-- meant a full-table rewrite of AgentGraphExecution setting NULL back to
-- NULL, which blew the platform's 2-minute statement timeout and failed the
-- whole deploy (SQLSTATE 57014). The startup backfill performs the real
-- assignment in bounded batches once orgs exist; reads stay null-tolerant
-- until then.

-- ==========================================================================
-- Step 2: (DEFERRED) Make organizationId NOT NULL
-- ==========================================================================
-- NOT NULL enforcement is deferred because:
--   1. The Prisma schema still declares these columns as nullable
--      (organizationId String?) so the generated client + writer code allows
--      NULL values. Setting NOT NULL on the DB column causes runtime
--      Null-constraint-violation errors when existing code paths and test
--      fixtures create rows without organizationId.
--   2. The cutover requires every caller (routes, executors, tests, seeders)
--      to provide organizationId — that work is in flight.
-- TODO(org-cutover-pr2): re-enable these ALTERs together with the
-- ``organizationId String`` schema change once all writers populate it.
--
-- ALTER TABLE "AgentGraph"          ALTER COLUMN "organizationId" SET NOT NULL;
-- ALTER TABLE "AgentGraphExecution" ALTER COLUMN "organizationId" SET NOT NULL;
-- ALTER TABLE "StoreListing"        ALTER COLUMN "owningOrgId" SET NOT NULL;

-- ==========================================================================
-- Step 3: (DEFERRED) Update StoreListing unique constraint
-- ==========================================================================
-- Depends on owningOrgId being NOT NULL — deferred along with Step 2.
-- TODO(org-cutover-pr2): drop the per-user slug uniqueness and add the
-- per-org one once owningOrgId is NOT NULL.
--
-- ALTER TABLE "StoreListing" DROP CONSTRAINT IF EXISTS "StoreListing_owningUserId_slug_key";
-- ALTER TABLE "StoreListing" ADD  CONSTRAINT "StoreListing_owningOrgId_slug_key" UNIQUE ("owningOrgId", slug);

-- ==========================================================================
-- Step 4: Recreate views with org-aware columns
-- ==========================================================================

-- 4a. Drop dependent views
DROP VIEW IF EXISTS "StoreAgent";
DROP VIEW IF EXISTS "Creator";
DROP VIEW IF EXISTS "StoreCreator";
DROP VIEW IF EXISTS "StoreSubmission";

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


-- 4c. Create StoreCreator view (replaces Creator, adds org_name)
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
    COALESCE(cs.agent_runs, 0::numeric)          AS agent_runs,
    o.name                                       AS org_name
FROM "Profile" p
LEFT JOIN creator_stats cs ON cs."owningUserId" = p."userId"
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

-- End of migration (Prisma handles transaction commit).
