-- Drop the trigger that auto-creates User + Profile on auth.users INSERT.
-- The invite activation flow in get_or_activate_user() now handles this.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'auth' AND table_name = 'users'
    ) THEN
        DROP TRIGGER IF EXISTS user_added_to_platform ON auth.users;
    END IF;
END $$;

DROP FUNCTION IF EXISTS add_user_and_profile_to_platform();
DROP FUNCTION IF EXISTS add_user_to_platform();
-- Keep generate_username() — used by backfill migration 20250205110132
