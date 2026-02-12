-- This migration creates a materialized view for suggested blocks based on execution counts
-- The view aggregates execution counts per block for the last 14 days
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
--   SET search_path TO platform;
--   SELECT refresh_suggested_blocks_view();

-- Check if pg_cron extension is installed
DO $$
DECLARE
    has_pg_cron BOOLEAN;
BEGIN
    SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_cron') INTO has_pg_cron;

    IF NOT has_pg_cron THEN
        RAISE WARNING 'pg_cron is not installed. Materialized view will be created but will NOT refresh automatically. For production, install pg_cron. For development, manually refresh with: SELECT refresh_suggested_blocks_view();';
    END IF;
END
$$;

-- Create materialized view for suggested blocks based on execution counts in last 14 days
-- The 14-day threshold is hardcoded to ensure consistent behavior
CREATE MATERIALIZED VIEW IF NOT EXISTS "mv_suggested_blocks" AS
SELECT
    agent_node."agentBlockId" AS block_id,
    COUNT(execution.id) AS execution_count
FROM "AgentNodeExecution" execution
JOIN "AgentNode" agent_node ON execution."agentNodeId" = agent_node.id
WHERE execution."endedTime" >= (NOW() - INTERVAL '14 days')
GROUP BY agent_node."agentBlockId"
ORDER BY execution_count DESC;

-- Create unique index for concurrent refresh support
CREATE UNIQUE INDEX IF NOT EXISTS "idx_mv_suggested_blocks_block_id" ON "mv_suggested_blocks"("block_id");

-- Create refresh function
CREATE OR REPLACE FUNCTION refresh_suggested_blocks_view()
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    target_schema text := current_schema();
BEGIN
    -- Use CONCURRENTLY for better performance during refresh
    REFRESH MATERIALIZED VIEW CONCURRENTLY "mv_suggested_blocks";
    RAISE NOTICE 'Suggested blocks materialized view refreshed in schema % at %', target_schema, NOW();
EXCEPTION
    WHEN OTHERS THEN
        -- Fallback to non-concurrent refresh if concurrent fails
        REFRESH MATERIALIZED VIEW "mv_suggested_blocks";
        RAISE NOTICE 'Suggested blocks materialized view refreshed (non-concurrent) in schema % at %. Concurrent refresh failed due to: %', target_schema, NOW(), SQLERRM;
END;
$$;

-- Initial refresh of the materialized view
SELECT refresh_suggested_blocks_view();

-- Schedule automatic refresh every hour (only if pg_cron is available)
DO $$
DECLARE
    has_pg_cron BOOLEAN;
    current_schema_name text := current_schema();
    job_name text;
BEGIN
    -- Check if pg_cron extension exists
    SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_cron') INTO has_pg_cron;

    IF has_pg_cron THEN
        job_name := format('refresh-suggested-blocks_%s', current_schema_name);

        -- Try to unschedule existing job (ignore errors if it doesn't exist)
        BEGIN
            PERFORM cron.unschedule(job_name);
        EXCEPTION WHEN OTHERS THEN
            NULL;
        END;

        -- Schedule the new job to run every hour
        PERFORM cron.schedule(
            job_name,
            '0 * * * *',  -- Every hour at minute 0
            format('SET search_path TO %I; SELECT refresh_suggested_blocks_view();', current_schema_name)
        );
        RAISE NOTICE 'Scheduled job %; runs every hour for schema %', job_name, current_schema_name;
    ELSE
        RAISE WARNING 'Automatic refresh NOT configured - pg_cron is not available. Manually refresh with: SELECT refresh_suggested_blocks_view();';
    END IF;
END;
$$;
