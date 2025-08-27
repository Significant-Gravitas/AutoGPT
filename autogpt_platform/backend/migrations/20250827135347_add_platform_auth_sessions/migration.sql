-- Migration: Add platform.auth_sessions table for long-term storage
-- This creates a copy of auth.sessions in the platform schema for permanent storage
-- since auth.sessions only retains data for ~1 month
-- This migration is idempotent and can be run multiple times safely
--
-- Features:
-- 1. Creates auth_sessions table in platform schema for permanent storage
-- 2. Automatically seeds all existing sessions from auth.sessions on first run
-- 3. Creates triggers to automatically copy new/updated sessions going forward
-- 4. Uses UPSERT (ON CONFLICT DO UPDATE) to ensure data consistency

-- Create the platform.auth_sessions table if it doesn't exist
-- This mirrors the structure of auth.sessions but in the platform schema for long-term storage
-- Using TEXT for aal column to avoid enum type dependency issues
CREATE TABLE IF NOT EXISTS "auth_sessions" (
  "id" uuid PRIMARY KEY,
  "user_id" uuid NOT NULL,
  "created_at" TIMESTAMP WITH TIME ZONE DEFAULT now(),
  "updated_at" TIMESTAMP WITH TIME ZONE DEFAULT now(),
  "factor_id" uuid,
  "aal" TEXT,
  "not_after" TIMESTAMP WITH TIME ZONE,
  "refreshed_at" TIMESTAMP WITH TIME ZONE,
  "user_agent" TEXT,
  "ip" inet,
  "tag" TEXT
);

-- Add indexes for performance (IF NOT EXISTS)
CREATE INDEX IF NOT EXISTS "idx_auth_sessions_user_id" ON "auth_sessions" ("user_id");
CREATE INDEX IF NOT EXISTS "idx_auth_sessions_created_at" ON "auth_sessions" ("created_at");
CREATE INDEX IF NOT EXISTS "idx_auth_sessions_updated_at" ON "auth_sessions" ("updated_at");

-- Create trigger function to copy data from auth.sessions to platform.auth_sessions
-- This function will only be created if the auth schema exists
DO $$
DECLARE
    v_row_count INTEGER;
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name = 'auth') THEN
        CREATE OR REPLACE FUNCTION copy_auth_session_to_platform()
        RETURNS TRIGGER AS $func$
        BEGIN
          -- Insert or update the session in the platform schema
          INSERT INTO "auth_sessions" (
            "id",
            "user_id", 
            "created_at",
            "updated_at",
            "factor_id",
            "aal",
            "not_after",
            "refreshed_at",
            "user_agent",
            "ip",
            "tag"
          ) VALUES (
            NEW.id,
            NEW.user_id,
            NEW.created_at,
            NEW.updated_at,
            NEW.factor_id,
            NEW.aal::text,
            NEW.not_after,
            NEW.refreshed_at,
            NEW.user_agent,
            NEW.ip,
            NEW.tag
          )
          ON CONFLICT (id) 
          DO UPDATE SET
            "user_id" = NEW.user_id,
            "updated_at" = NEW.updated_at,
            "factor_id" = NEW.factor_id,
            "aal" = NEW.aal::text,
            "not_after" = NEW.not_after,
            "refreshed_at" = NEW.refreshed_at,
            "user_agent" = NEW.user_agent,
            "ip" = NEW.ip,
            "tag" = NEW.tag;
          
          RETURN NEW;
        END;
        $func$ LANGUAGE plpgsql;

        -- Create triggers on auth.sessions table if it exists
        IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'auth' AND table_name = 'sessions') THEN
            DROP TRIGGER IF EXISTS trigger_copy_auth_session_insert ON "auth"."sessions";
            DROP TRIGGER IF EXISTS trigger_copy_auth_session_update ON "auth"."sessions";
            
            CREATE TRIGGER trigger_copy_auth_session_insert
              AFTER INSERT ON "auth"."sessions"
              FOR EACH ROW
              EXECUTE FUNCTION copy_auth_session_to_platform();

            CREATE TRIGGER trigger_copy_auth_session_update
              AFTER UPDATE ON "auth"."sessions"
              FOR EACH ROW
              EXECUTE FUNCTION copy_auth_session_to_platform();

            -- Initial seeding: Copy ALL existing data from auth.sessions to platform.auth_sessions
            -- This ensures we don't lose any historical session data
            
            -- Check current state
            SELECT COUNT(*) INTO v_row_count FROM "auth_sessions";
            RAISE NOTICE 'Current auth_sessions table has % existing records', v_row_count;
            
            SELECT COUNT(*) INTO v_row_count FROM "auth"."sessions";
            RAISE NOTICE 'Found % records in auth.sessions to seed', v_row_count;
            
            -- Perform the seeding with UPSERT to handle existing records
            INSERT INTO "auth_sessions" (
              "id",
              "user_id", 
              "created_at",
              "updated_at",
              "factor_id",
              "aal",
              "not_after",
              "refreshed_at",
              "user_agent",
              "ip",
              "tag"
            )
            SELECT 
              "id",
              "user_id",
              "created_at",
              "updated_at",
              "factor_id",
              "aal"::text,
              "not_after",
              "refreshed_at",
              "user_agent",
              "ip",
              "tag"
            FROM "auth"."sessions"
            ON CONFLICT (id) DO UPDATE SET
              "user_id" = EXCLUDED.user_id,
              "updated_at" = EXCLUDED.updated_at,
              "factor_id" = EXCLUDED.factor_id,
              "aal" = EXCLUDED.aal,
              "not_after" = EXCLUDED.not_after,
              "refreshed_at" = EXCLUDED.refreshed_at,
              "user_agent" = EXCLUDED.user_agent,
              "ip" = EXCLUDED.ip,
              "tag" = EXCLUDED.tag;
            
            GET DIAGNOSTICS v_row_count = ROW_COUNT;
            RAISE NOTICE 'Successfully processed % session records', v_row_count;
            
            -- Final count
            SELECT COUNT(*) INTO v_row_count FROM "auth_sessions";
            RAISE NOTICE 'auth_sessions table now contains % total records', v_row_count;
        END IF;
    END IF;
END $$;