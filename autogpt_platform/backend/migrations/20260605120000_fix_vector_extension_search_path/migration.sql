-- Make the pgvector `vector` type resolvable on DIRECT (non-pooled) connections.
--
-- Background:
--   The previous migration (20260129150000) moved pgvector into the `extensions`
--   schema and set a role-level search_path of:
--       "$user", platform, public, extensions
--   That works on POOLED (Supavisor) connections used by dev/prod: the pooler
--   ignores the per-connection `?schema=platform` parameter and falls back to the
--   role-level search_path, which includes `extensions`.
--
--   It does NOT work on DIRECT connections (local dev, CI, self-hosted without
--   Supavisor). A direct connection honors `?schema=platform`, which sets a
--   SESSION-level `search_path = "platform"`. A session-level search_path overrides
--   the role-level one, so `extensions` is not visible and unqualified `::vector`
--   casts fail with: type "vector" does not exist.
--
-- Fix:
--   Move the `vector` extension into the application schema (`platform`), the only
--   schema guaranteed to be on the search_path of BOTH connection types (direct:
--   forced by `?schema=platform`; pooled: included in the role-level search_path).
--   The role-level search_path set by the previous migration already covers pooled
--   connections, so it does not need to be touched here.
DO $$
DECLARE
    app_schema text := current_schema();
    vector_schema text;
BEGIN
    SELECT n.nspname INTO vector_schema
    FROM pg_extension e
    JOIN pg_namespace n ON e.extnamespace = n.oid
    WHERE e.extname = 'vector';

    IF vector_schema IS NULL THEN
        -- Not installed yet: create it directly in the application schema.
        EXECUTE format('CREATE EXTENSION IF NOT EXISTS vector SCHEMA %I', app_schema);
    ELSIF vector_schema <> app_schema THEN
        -- Move it into the application schema. May fail on managed platforms (e.g.
        -- Supabase) where the extension is owned by a system role; that is fine,
        -- because those use pooled connections that resolve the type via the
        -- role-level search_path established by migration 20260129150000.
        BEGIN
            RAISE NOTICE 'Moving pgvector from schema "%" to "%"', vector_schema, app_schema;
            EXECUTE format('ALTER EXTENSION vector SET SCHEMA %I', app_schema);
        EXCEPTION WHEN insufficient_privilege THEN
            RAISE NOTICE 'Cannot move pgvector into "%" (insufficient privileges); '
                'pooled connections will resolve it via the role-level search_path.', app_schema;
        END;
    END IF;
END $$;
