-- =============================================================
-- AutoGPT Analytics Schema Setup
-- Run ONCE in Supabase SQL Editor as the postgres superuser.
-- After this, run generate_views.py to create/refresh the views.
-- =============================================================

-- 1. Create the analytics schema
CREATE SCHEMA IF NOT EXISTS analytics;

-- 2. Create the read-only role (skip if already exists)
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'analytics_readonly') THEN
    CREATE ROLE analytics_readonly WITH LOGIN PASSWORD 'CHANGE_ME';
  END IF;
END
$$;

-- 3. Auth schema grants
--    Supabase restricts the auth schema; run as postgres superuser.
GRANT USAGE ON SCHEMA auth TO analytics_readonly;
GRANT SELECT ON auth.sessions TO analytics_readonly;
GRANT SELECT ON auth.audit_log_entries TO analytics_readonly;

-- 4. Platform schema grants
GRANT USAGE ON SCHEMA platform TO analytics_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA platform TO analytics_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA platform
  GRANT SELECT ON TABLES TO analytics_readonly;

-- 5. Analytics schema grants
GRANT USAGE ON SCHEMA analytics TO analytics_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO analytics_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA analytics
  GRANT SELECT ON TABLES TO analytics_readonly;
