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

-- Step 2: Move pgvector from current schema to extensions schema
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
        -- Installed in wrong schema; move it
        RAISE NOTICE 'Moving pgvector from schema "%" to "extensions"', vector_schema;
        ALTER EXTENSION vector SET SCHEMA extensions;
    ELSE
        RAISE NOTICE 'pgvector already in extensions schema, nothing to do';
    END IF;
END $$;

-- Step 3: Update database default search_path to include the application schema
-- and extensions schema. This is a permanent database-level setting that applies
-- to all new connections.
-- The default PostgreSQL search_path is: "$user", public
-- We add the current application schema and extensions so that:
--   - ::vector type casts resolve (via extensions schema)
--   - Application tables/types resolve in raw queries (via application schema)
DO $$
DECLARE
    app_schema text;
BEGIN
    SELECT current_schema() INTO app_schema;

    IF app_schema = 'public' THEN
        -- In CI/local with public schema, extensions is the only addition needed
        EXECUTE format(
            'ALTER DATABASE %I SET search_path TO "$user", public, extensions',
            current_database()
        );
    ELSE
        -- In production with a custom schema (e.g. platform), include it
        EXECUTE format(
            'ALTER DATABASE %I SET search_path TO "$user", public, extensions, %I',
            current_database(), app_schema
        );
    END IF;
END $$;
