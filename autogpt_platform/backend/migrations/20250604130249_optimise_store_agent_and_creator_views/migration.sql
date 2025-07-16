-- This migration creates materialized views for performance optimization
-- 
-- IMPORTANT: For production environments, pg_cron is REQUIRED for automatic refresh
-- Prerequisites for production:
--   1. pg_cron extension must be installed: CREATE EXTENSION pg_cron;
--   2. pg_cron must be configured in postgresql.conf:
--      shared_preload_libraries = 'pg_cron'
--      cron.database_name = 'your_database_name'
--
-- For development environments without pg_cron:
--   The migration will succeed but you must manually refresh views with:
--   SELECT refresh_store_materialized_views();

-- Check if pg_cron extension is installed and set a flag
DO $$
DECLARE
    has_pg_cron BOOLEAN;
BEGIN
    SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_cron') INTO has_pg_cron;
    
    IF NOT has_pg_cron THEN
        RAISE WARNING 'pg_cron extension is not installed!';
        RAISE WARNING 'Materialized views will be created but WILL NOT refresh automatically.';
        RAISE WARNING 'For production use, install pg_cron with: CREATE EXTENSION pg_cron;';
        RAISE WARNING 'For development, manually refresh with: SELECT refresh_store_materialized_views();';
        
        -- For production deployments, uncomment the following line to make pg_cron mandatory:
        -- RAISE EXCEPTION 'pg_cron is required for production deployments';
    END IF;
    
    -- Store the flag for later use in the migration
    PERFORM set_config('migration.has_pg_cron', has_pg_cron::text, false);
END
$$;

-- CreateIndex
-- Optimized: Only include owningUserId in index columns since isDeleted and hasApprovedVersion are in WHERE clause
CREATE INDEX IF NOT EXISTS "idx_store_listing_approved" ON "StoreListing"("owningUserId") WHERE "isDeleted" = false AND "hasApprovedVersion" = true;

-- CreateIndex
-- Optimized: Only include storeListingId since submissionStatus is in WHERE clause
CREATE INDEX IF NOT EXISTS "idx_store_listing_version_status" ON "StoreListingVersion"("storeListingId") WHERE "submissionStatus" = 'APPROVED';

-- CreateIndex
CREATE INDEX IF NOT EXISTS "idx_slv_categories_gin" ON "StoreListingVersion" USING GIN ("categories") WHERE "submissionStatus" = 'APPROVED';

-- CreateIndex
CREATE INDEX IF NOT EXISTS "idx_slv_agent" ON "StoreListingVersion"("agentGraphId", "agentGraphVersion") WHERE "submissionStatus" = 'APPROVED';

-- CreateIndex
CREATE INDEX IF NOT EXISTS "idx_store_listing_review_version" ON "StoreListingReview"("storeListingVersionId");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "idx_agent_graph_execution_agent" ON "AgentGraphExecution"("agentGraphId");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "idx_profile_user" ON "Profile"("userId");

-- Additional performance indexes
CREATE INDEX IF NOT EXISTS "idx_store_listing_version_approved_listing" ON "StoreListingVersion"("storeListingId", "version") WHERE "submissionStatus" = 'APPROVED';

-- Create materialized view for agent run counts
CREATE MATERIALIZED VIEW IF NOT EXISTS "mv_agent_run_counts" AS
SELECT 
    "agentGraphId", 
    COUNT(*) AS run_count
FROM "AgentGraphExecution"
GROUP BY "agentGraphId";

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "idx_mv_agent_run_counts" ON "mv_agent_run_counts"("agentGraphId");

-- Create materialized view for review statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS "mv_review_stats" AS
SELECT 
    sl.id AS "storeListingId",
    COUNT(sr.id) AS review_count,
    AVG(sr.score::numeric) AS avg_rating
FROM "StoreListing" sl
JOIN "StoreListingVersion" slv ON slv."storeListingId" = sl.id
LEFT JOIN "StoreListingReview" sr ON sr."storeListingVersionId" = slv.id
WHERE sl."isDeleted" = false
  AND slv."submissionStatus" = 'APPROVED'
GROUP BY sl.id;

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "idx_mv_review_stats" ON "mv_review_stats"("storeListingId");

-- DropForeignKey (if any exist on the views)
-- None needed as views don't have foreign keys

-- DropView
DROP VIEW IF EXISTS "Creator";

-- DropView
DROP VIEW IF EXISTS "StoreAgent";

-- CreateView
CREATE OR REPLACE VIEW "StoreAgent" AS
WITH agent_versions AS (
    SELECT 
        "storeListingId",
        array_agg(DISTINCT version::text ORDER BY version::text) AS versions
    FROM "StoreListingVersion"
    WHERE "submissionStatus" = 'APPROVED'
    GROUP BY "storeListingId"
)
SELECT
    sl.id AS listing_id,
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
    COALESCE(av.versions, ARRAY[slv.version::text]) AS versions
FROM "StoreListing" sl
INNER JOIN "StoreListingVersion" slv 
    ON slv."storeListingId" = sl.id 
    AND slv."submissionStatus" = 'APPROVED'
JOIN "AgentGraph" a 
    ON slv."agentGraphId" = a.id 
    AND slv."agentGraphVersion" = a.version
LEFT JOIN "Profile" p 
    ON sl."owningUserId" = p."userId"
LEFT JOIN "mv_review_stats" rs 
    ON sl.id = rs."storeListingId"
LEFT JOIN "mv_agent_run_counts" ar 
    ON a.id = ar."agentGraphId"
LEFT JOIN agent_versions av 
    ON sl.id = av."storeListingId"
WHERE sl."isDeleted" = false
  AND sl."hasApprovedVersion" = true;

-- CreateView
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
    INNER JOIN "StoreListingVersion" slv 
        ON slv."storeListingId" = sl.id 
        AND slv."submissionStatus" = 'APPROVED'
    LEFT JOIN "StoreListingReview" sr 
        ON sr."storeListingVersionId" = slv.id
    LEFT JOIN "mv_agent_run_counts" ar 
        ON ar."agentGraphId" = slv."agentGraphId"
    WHERE sl."isDeleted" = false 
      AND sl."hasApprovedVersion" = true
),
creator_stats AS (
    SELECT 
        cl."owningUserId",
        COUNT(DISTINCT cl.listing_id) AS num_agents,
        AVG(COALESCE(cl.score, 0)::numeric) AS agent_rating,
        SUM(DISTINCT COALESCE(cl.run_count, 0)) AS agent_runs,
        array_agg(DISTINCT cat ORDER BY cat) FILTER (WHERE cat IS NOT NULL) AS all_categories
    FROM creator_listings cl
    LEFT JOIN LATERAL unnest(COALESCE(cl.categories, ARRAY[]::text[])) AS cat ON true
    GROUP BY cl."owningUserId"
)
SELECT 
    p.username,
    p.name,
    p."avatarUrl" AS avatar_url,
    p.description,
    cs.all_categories AS top_categories,
    p.links,
    p."isFeatured" AS is_featured,
    COALESCE(cs.num_agents, 0::bigint) AS num_agents,
    COALESCE(cs.agent_rating, 0.0) AS agent_rating,
    COALESCE(cs.agent_runs, 0::numeric) AS agent_runs
FROM "Profile" p
LEFT JOIN creator_stats cs ON cs."owningUserId" = p."userId";

-- Create refresh function that works with the current schema
CREATE OR REPLACE FUNCTION refresh_store_materialized_views()
RETURNS void 
LANGUAGE plpgsql
AS $$
DECLARE
    current_schema_name text;
BEGIN
    -- Get the current schema
    current_schema_name := current_schema();
    
    -- Use CONCURRENTLY for better performance during refresh
    EXECUTE format('REFRESH MATERIALIZED VIEW CONCURRENTLY %I."mv_agent_run_counts"', current_schema_name);
    EXECUTE format('REFRESH MATERIALIZED VIEW CONCURRENTLY %I."mv_review_stats"', current_schema_name);
    RAISE NOTICE 'Materialized views refreshed in schema % at %', current_schema_name, NOW();
EXCEPTION 
    WHEN OTHERS THEN
        -- Fallback to non-concurrent refresh if concurrent fails
        EXECUTE format('REFRESH MATERIALIZED VIEW %I."mv_agent_run_counts"', current_schema_name);
        EXECUTE format('REFRESH MATERIALIZED VIEW %I."mv_review_stats"', current_schema_name);
        RAISE NOTICE 'Materialized views refreshed (non-concurrent) in schema % at % due to: %', current_schema_name, NOW(), SQLERRM;
END;
$$;

-- Initial refresh of materialized views
SELECT refresh_store_materialized_views();

-- Schedule automatic refresh every 15 minutes (only if pg_cron is available)
DO $$
DECLARE
    has_pg_cron BOOLEAN;
    current_schema_name text;
    job_name text;
BEGIN
    -- Get the flag we set earlier
    has_pg_cron := current_setting('migration.has_pg_cron', true)::boolean;
    
    -- Get current schema name
    current_schema_name := current_schema();
    
    -- Create a unique job name for this schema
    job_name := format('refresh-store-views-%s', current_schema_name);
    
    IF has_pg_cron THEN
        -- Try to unschedule existing job (ignore errors if it doesn't exist)
        BEGIN
            PERFORM cron.unschedule(job_name);
        EXCEPTION WHEN OTHERS THEN
            -- Job doesn't exist, that's fine
            NULL;
        END;
        
        -- Schedule the refresh job with schema-specific command
        PERFORM cron.schedule(
            job_name,
            '*/15 * * * *',
            format('SELECT %I.refresh_store_materialized_views();', current_schema_name)
        );
        RAISE NOTICE 'Scheduled automatic refresh of materialized views every 15 minutes for schema %', current_schema_name;
    ELSE
        RAISE WARNING '⚠️  Automatic refresh NOT configured - pg_cron is not available';
        RAISE WARNING '⚠️  You must manually refresh views with: SELECT refresh_store_materialized_views();';
        RAISE WARNING '⚠️  Or install pg_cron for automatic refresh in production';
    END IF;
END;
$$;