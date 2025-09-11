-- Fixes the refresh function+job introduced in 20250604130249_optimise_store_agent_and_creator_views
-- by improving the function to accept a schema parameter and updating the cron job to use it.
-- This resolves the issue where pg_cron jobs fail because they run in 'public' schema
-- but the materialized views exist in 'platform' schema.


-- Drop the old function first to avoid ambiguity
DROP FUNCTION IF EXISTS refresh_store_materialized_views();

-- Create parameterized refresh function that accepts schema name
CREATE OR REPLACE FUNCTION refresh_store_materialized_views(target_schema text DEFAULT current_schema())
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
    -- Use CONCURRENTLY for better performance during refresh
    EXECUTE format('REFRESH MATERIALIZED VIEW CONCURRENTLY %I."mv_agent_run_counts"', target_schema);
    EXECUTE format('REFRESH MATERIALIZED VIEW CONCURRENTLY %I."mv_review_stats"', target_schema);
    RAISE NOTICE 'Materialized views refreshed in schema % at %', target_schema, NOW();
EXCEPTION
    WHEN OTHERS THEN
        -- Fallback to non-concurrent refresh if concurrent fails
        EXECUTE format('REFRESH MATERIALIZED VIEW %I."mv_agent_run_counts"', target_schema);
        EXECUTE format('REFRESH MATERIALIZED VIEW %I."mv_review_stats"', target_schema);
        RAISE NOTICE 'Materialized views refreshed (non-concurrent) in schema % at %. Concurrent refresh failed due to: %', target_schema, NOW(), SQLERRM;
END;
$$;

-- Initial refresh + test of the function to ensure it works
SELECT refresh_store_materialized_views();

-- Re-create the cron job to use the improved function
DO $$
DECLARE
    has_pg_cron BOOLEAN;
    current_schema_name text;
    old_job_name text;
    job_name text;
BEGIN
    -- Check if pg_cron extension exists
    SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_cron') INTO has_pg_cron;

    IF has_pg_cron THEN
        -- Get current schema for unique job naming
        current_schema_name := current_schema();
        old_job_name := format('refresh-store-views-%s', current_schema_name);
        job_name := format('refresh-store-views_%s', current_schema_name);

        -- Try to unschedule existing job (ignore errors if it doesn't exist)
        BEGIN
            PERFORM cron.unschedule(old_job_name);
        EXCEPTION WHEN OTHERS THEN
            NULL;
        END;

        -- Schedule the new job with explicit schema parameter
        PERFORM cron.schedule(
            job_name,
            '*/15 * * * *',
            format('SELECT %I.refresh_store_materialized_views(%L);', current_schema_name, current_schema_name)
        );
        RAISE NOTICE 'Scheduled job %; runs every 15 minutes for schema %', job_name, current_schema_name;
    ELSE
        RAISE WARNING '⚠️  Automatic refresh NOT configured - pg_cron is not available';
        RAISE WARNING '⚠️  You must manually refresh views with: SELECT refresh_store_materialized_views();';
        RAISE WARNING '⚠️  Or install pg_cron for automatic refresh in production';
    END IF;
END;
$$;
