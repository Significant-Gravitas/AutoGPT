-- Unschedule cron job (if it exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_cron') THEN
        PERFORM cron.unschedule('refresh-store-views');
        RAISE NOTICE 'Unscheduled automatic refresh of materialized views';
    END IF;
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Could not unschedule cron job (may not exist): %', SQLERRM;
END;
$$;

-- DropView
DROP VIEW IF EXISTS "Creator";

-- DropView  
DROP VIEW IF EXISTS "StoreAgent";

-- CreateView (restore original StoreAgent)
CREATE VIEW "StoreAgent" AS
WITH reviewstats AS (
    SELECT sl_1.id AS "storeListingId",
           count(sr.id) AS review_count,
           avg(sr.score::numeric) AS avg_rating
      FROM "StoreListing" sl_1
      JOIN "StoreListingVersion" slv_1
           ON slv_1."storeListingId" = sl_1.id
      JOIN "StoreListingReview" sr
           ON sr."storeListingVersionId" = slv_1.id
     WHERE sl_1."isDeleted" = false
     GROUP BY sl_1.id
), agentruns AS (
    SELECT "AgentGraphExecution"."agentGraphId",
           count(*) AS run_count
      FROM "AgentGraphExecution"
     GROUP BY "AgentGraphExecution"."agentGraphId"
)
SELECT sl.id AS listing_id,
       slv.id AS "storeListingVersionId",
       slv."createdAt" AS updated_at,
       sl.slug,
       COALESCE(slv.name, '') AS agent_name,
       slv."videoUrl" AS agent_video,
       COALESCE(slv."imageUrls", ARRAY[]::text[]) AS agent_image,
       slv."isFeatured" AS featured,
       p.username AS creator_username,
       p."avatarUrl" AS creator_avatar,
       slv."subHeading" AS sub_heading,
       slv.description,
       slv.categories,
       COALESCE(ar.run_count, 0::bigint) AS runs,
       COALESCE(rs.avg_rating, 0.0)::double precision AS rating,
       array_agg(DISTINCT slv.version::text) AS versions
  FROM "StoreListing" sl
  JOIN "StoreListingVersion" slv
       ON slv."storeListingId" = sl.id
  JOIN "AgentGraph" a
       ON slv."agentGraphId" = a.id
      AND slv."agentGraphVersion" = a.version
  LEFT JOIN "Profile" p
       ON sl."owningUserId" = p."userId"
  LEFT JOIN reviewstats rs
       ON sl.id = rs."storeListingId"
  LEFT JOIN agentruns ar
       ON a.id = ar."agentGraphId"
 WHERE sl."isDeleted" = false
   AND sl."hasApprovedVersion" = true
   AND slv."submissionStatus" = 'APPROVED'
 GROUP BY sl.id, slv.id, sl.slug, slv."createdAt", slv.name, slv."videoUrl",
          slv."imageUrls", slv."isFeatured", p.username, p."avatarUrl",
          slv."subHeading", slv.description, slv.categories, ar.run_count,
          rs.avg_rating;

-- CreateView (restore original Creator)
CREATE VIEW "Creator" AS
WITH agentstats AS (
    SELECT p_1.username,
           count(DISTINCT sl.id) AS num_agents,
           avg(COALESCE(sr.score, 0)::numeric) AS agent_rating,
           sum(COALESCE(age.run_count, 0::bigint)) AS agent_runs
      FROM "Profile" p_1
 LEFT JOIN "StoreListing" sl
        ON sl."owningUserId" = p_1."userId"
 LEFT JOIN "StoreListingVersion" slv
        ON slv."storeListingId" = sl.id
 LEFT JOIN "StoreListingReview" sr
        ON sr."storeListingVersionId" = slv.id
 LEFT JOIN (
          SELECT "AgentGraphExecution"."agentGraphId",
                 count(*) AS run_count
            FROM "AgentGraphExecution"
        GROUP BY "AgentGraphExecution"."agentGraphId"
      ) age ON age."agentGraphId" = slv."agentGraphId"
     WHERE sl."isDeleted" = false
       AND sl."hasApprovedVersion" = true
       AND slv."submissionStatus" = 'APPROVED'
  GROUP BY p_1.username
)
SELECT p.username,
       p.name,
       p."avatarUrl" AS avatar_url,
       p.description,
       array_agg(DISTINCT cats.c) FILTER (WHERE cats.c IS NOT NULL) AS top_categories,
       p.links,
       p."isFeatured" AS is_featured,
       COALESCE(ast.num_agents, 0::bigint) AS num_agents,
       COALESCE(ast.agent_rating, 0.0) AS agent_rating,
       COALESCE(ast.agent_runs, 0::numeric) AS agent_runs
  FROM "Profile" p
  LEFT JOIN agentstats ast
         ON ast.username = p.username
  LEFT JOIN LATERAL (
          SELECT unnest(slv.categories) AS c
            FROM "StoreListing" sl
            JOIN "StoreListingVersion" slv
                 ON slv."storeListingId" = sl.id
           WHERE sl."owningUserId" = p."userId"
             AND sl."isDeleted" = false
             AND sl."hasApprovedVersion" = true
             AND slv."submissionStatus" = 'APPROVED'
       ) cats ON true
 GROUP BY p.username, p.name, p."avatarUrl", p.description, p.links,
          p."isFeatured", ast.num_agents, ast.agent_rating, ast.agent_runs;

-- Drop function
DROP FUNCTION IF EXISTS platform.refresh_store_materialized_views();

-- Drop materialized views
DROP MATERIALIZED VIEW IF EXISTS "mv_review_stats";
DROP MATERIALIZED VIEW IF EXISTS "mv_agent_run_counts";

-- DropIndex
DROP INDEX IF EXISTS "idx_profile_user";

-- DropIndex
DROP INDEX IF EXISTS "idx_agent_graph_execution_agent";

-- DropIndex
DROP INDEX IF EXISTS "idx_store_listing_review_version";

-- DropIndex
DROP INDEX IF EXISTS "idx_slv_agent";

-- DropIndex
DROP INDEX IF EXISTS "idx_slv_categories_gin";

-- DropIndex
DROP INDEX IF EXISTS "idx_store_listing_version_status";

-- DropIndex
DROP INDEX IF EXISTS "idx_store_listing_approved";

-- DropIndex  
DROP INDEX IF EXISTS "idx_store_listing_version_approved_listing";