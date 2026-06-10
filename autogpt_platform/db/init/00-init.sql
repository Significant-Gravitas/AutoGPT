-- Initializes a fresh local Postgres for the AutoGPT platform.
-- Runs once via /docker-entrypoint-initdb.d/ when the data volume is empty.
--
-- Auth is handled by Better Auth (embedded in the Next.js frontend); its
-- tables live in the `platform` schema and are created by the backend's
-- Prisma migrations. Nothing Supabase-related runs in this stack anymore.

-- Schema used by Prisma (DATABASE_URL ?schema=platform) and Better Auth.
CREATE SCHEMA IF NOT EXISTS platform;

-- Legacy `auth` schema shim.
--
-- Historical Prisma migrations reference auth.users (triggers, backfills):
--   20241007175112_add_oauth_creds_user_trigger
--   20250205100104_add_profile_trigger
--   20260311000000_drop_auto_user_trigger
--   20260319120000_revert_invite_system
--   20260610120000_add_better_auth_tables
-- All of them guard on the table's existence, but the revert_invite_system
-- backfill and the trigger re-creation SELECT real columns from auth.users.
-- This empty shim lets the full migration history apply cleanly on a fresh
-- database. It is never written to at runtime.
CREATE SCHEMA IF NOT EXISTS auth;

CREATE TABLE IF NOT EXISTS auth.users (
    id uuid PRIMARY KEY,
    email text,
    encrypted_password text,
    raw_user_meta_data jsonb DEFAULT '{}',
    raw_app_meta_data jsonb DEFAULT '{}',
    role text,
    is_super_admin boolean,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now(),
    deleted_at timestamptz,
    email_confirmed_at timestamptz,
    phone text,
    banned_until timestamptz
);

-- No other Supabase objects (supabase_functions, realtime, storage, pgsodium,
-- vault, ...) are referenced by the migration history. The pgvector and
-- pg_trgm extensions are created by the migrations themselves and ship with
-- the pgvector/pgvector image.
