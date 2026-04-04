-- Move pgvector extension from application schema to extensions schema.
-- The extensions schema is in the default search_path, so ::vector casts
-- in raw SQL queries will resolve correctly without needing to set
-- search_path per connection. This fixes intermittent "type vector does not exist"
-- errors that occur when pooled connections don't have the right search_path.

-- Also set the database default search_path to include the application schema,
-- so that raw SQL queries can find application tables/types without explicit
-- schema prefix.

-- Step 1: Ensure extensions schema exists
CREATE SCHEMA IF NOT EXISTS extensions;

-- Step 2: Move pgvector from current schema to extensions schema (if possible)
DO $$
DECLARE
    vector_schema text;
BEGIN
    -- Check where vector extension currently lives
    SELECT n.nspname INTO vector_schema
    FROM pg_extension e
    JOIN pg_namespace n ON e.extnamespace = n.oid
    WHERE e.extname = 'vector';

    IF vector_schema IS NULL THEN
        -- Not installed yet; create in extensions schema
        CREATE EXTENSION IF NOT EXISTS vector SCHEMA extensions;
    ELSIF vector_schema != 'extensions' THEN
        -- Installed in wrong schema; try to move it
        -- This may fail on managed platforms like Supabase where the extension
        -- is owned by a system user. In that case, we'll include the current
        -- vector schema in the search_path instead (handled in Step 3).
        BEGIN
            RAISE NOTICE 'Moving pgvector from schema "%" to "extensions"', vector_schema;
            ALTER EXTENSION vector SET SCHEMA extensions;
        EXCEPTION WHEN insufficient_privilege THEN
            RAISE NOTICE 'Cannot move pgvector (insufficient privileges). Will include schema "%" in search_path instead.', vector_schema;
        END;
    ELSE
        RAISE NOTICE 'pgvector already in extensions schema, nothing to do';
    END IF;
END $$;

-- Step 3: Update role-level search_path to include the application schema
-- and the schema where pgvector lives. Role-level settings take precedence over
-- database-level settings, which is important for managed platforms like Supabase
-- that may have existing role-level search_path configurations.
-- The default PostgreSQL search_path is: "$user", public
-- We add the extensions schema and (if different) the schema where pgvector
-- actually lives, so that ::vector type casts resolve correctly.
DO $$
DECLARE
    app_schema text := current_schema();
    vector_schema text;
    ext_schemas text := 'extensions';
BEGIN
    -- Find where pgvector currently lives (after Step 2's move-or-fail)
    SELECT n.nspname INTO vector_schema
    FROM pg_extension e
    JOIN pg_namespace n ON e.extnamespace = n.oid
    WHERE e.extname = 'vector';

    -- If vector is in a schema not already covered, add it to the search_path
    IF vector_schema IS NOT NULL
        AND vector_schema != 'extensions'
        AND vector_schema != 'public'
        AND vector_schema != app_schema
    THEN
        ext_schemas := format('extensions, %I', vector_schema);
    END IF;

    IF app_schema = 'public' THEN
        EXECUTE format(
            'ALTER ROLE %I SET search_path TO "$user", public, %s',
            current_user, ext_schemas
        );
    ELSE
        EXECUTE format(
            'ALTER ROLE %I SET search_path TO "$user", %I, public, %s',
            current_user, app_schema, ext_schemas
        );
    END IF;
END $$;
