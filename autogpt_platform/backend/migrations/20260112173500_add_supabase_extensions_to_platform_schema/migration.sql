-- Acknowledge Supabase-managed extensions to prevent drift warnings
-- These extensions are pre-installed by Supabase in specific schemas
-- This migration ensures they exist where available (Supabase) or skips gracefully (CI)

-- Create schemas (safe in both CI and Supabase)
CREATE SCHEMA IF NOT EXISTS "extensions";

-- Extensions that exist in both CI and Supabase
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS "pgcrypto" WITH SCHEMA "extensions";
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pgcrypto extension not available, skipping';
END $$;

DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA "extensions";
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'uuid-ossp extension not available, skipping';
END $$;

-- Supabase-specific extensions (skip gracefully in CI)
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS "pg_stat_statements" WITH SCHEMA "extensions";
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pg_stat_statements extension not available, skipping';
END $$;

DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS "pg_net" WITH SCHEMA "extensions";
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pg_net extension not available, skipping';
END $$;

DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS "pgjwt" WITH SCHEMA "extensions";
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pgjwt extension not available, skipping';
END $$;

DO $$
BEGIN
    CREATE SCHEMA IF NOT EXISTS "graphql";
    CREATE EXTENSION IF NOT EXISTS "pg_graphql" WITH SCHEMA "graphql";
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pg_graphql extension not available, skipping';
END $$;

DO $$
BEGIN
    CREATE SCHEMA IF NOT EXISTS "pgsodium";
    CREATE EXTENSION IF NOT EXISTS "pgsodium" WITH SCHEMA "pgsodium";
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pgsodium extension not available, skipping';
END $$;

DO $$
BEGIN
    CREATE SCHEMA IF NOT EXISTS "vault";
    CREATE EXTENSION IF NOT EXISTS "supabase_vault" WITH SCHEMA "vault";
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'supabase_vault extension not available, skipping';
END $$;


-- Return to platform
CREATE SCHEMA IF NOT EXISTS "platform";