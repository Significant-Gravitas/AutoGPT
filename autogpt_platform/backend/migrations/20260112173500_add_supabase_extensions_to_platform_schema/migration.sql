-- -- Acknowledge Supabase-managed extensions to prevent drift warnings
-- -- These extensions are pre-installed by Supabase in specific schemas
-- -- This migration just documents their existence for Prisma's migration history

-- -- Note: These schemas and extensions are created by Supabase, not by this migration
-- -- Using IF NOT EXISTS ensures this migration is safe to run multiple times

CREATE SCHEMA IF NOT EXISTS "extensions";
CREATE EXTENSION IF NOT EXISTS "pgcrypto" WITH SCHEMA "extensions";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements" WITH SCHEMA "extensions";
CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA "extensions";
CREATE EXTENSION IF NOT EXISTS "pg_net" WITH SCHEMA "extensions";
CREATE EXTENSION IF NOT EXISTS "pgjwt" WITH SCHEMA "extensions";

CREATE SCHEMA IF NOT EXISTS "graphql";
CREATE EXTENSION IF NOT EXISTS "pg_graphql" WITH SCHEMA "graphql";

CREATE SCHEMA IF NOT EXISTS "pgsodium";
CREATE EXTENSION IF NOT EXISTS "pgsodium" WITH SCHEMA "pgsodium";

CREATE SCHEMA IF NOT EXISTS "vault";
CREATE EXTENSION IF NOT EXISTS "supabase_vault" WITH SCHEMA "vault";

