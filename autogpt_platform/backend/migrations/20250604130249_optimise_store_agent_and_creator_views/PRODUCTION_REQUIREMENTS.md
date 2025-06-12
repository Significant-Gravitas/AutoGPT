# Production Requirements for Materialized Views Migration

This migration creates materialized views that require automatic refresh for optimal performance. 

## Required for Production

### 1. Install pg_cron Extension

```sql
-- As superuser or database owner:
CREATE EXTENSION IF NOT EXISTS pg_cron;
```

### 2. Configure PostgreSQL

Add to `postgresql.conf`:
```
shared_preload_libraries = 'pg_cron'
cron.database_name = 'your_database_name'
```

Then restart PostgreSQL.

### 3. Verify Installation

```sql
-- Check if pg_cron is installed
SELECT * FROM pg_extension WHERE extname = 'pg_cron';

-- Check if the refresh job is scheduled
SELECT * FROM cron.job WHERE jobname = 'refresh-store-views';
```

## Making pg_cron Mandatory

To enforce pg_cron requirement in production, modify line 28 in `migration.sql`:

```sql
-- Change from:
-- RAISE EXCEPTION 'pg_cron is required for production deployments';

-- To:
RAISE EXCEPTION 'pg_cron is required for production deployments';
```

## Manual Refresh (Development Only)

If pg_cron is not available in development:

```sql
-- Refresh materialized views manually
SELECT refresh_store_materialized_views();
```

## Monitoring

Check materialized view freshness:
```sql
-- Check when views were last refreshed
SELECT 
    schemaname,
    matviewname,
    last_refresh
FROM pg_stat_user_tables 
WHERE schemaname = 'public' 
AND tablename IN ('mv_agent_run_counts', 'mv_review_stats');
```