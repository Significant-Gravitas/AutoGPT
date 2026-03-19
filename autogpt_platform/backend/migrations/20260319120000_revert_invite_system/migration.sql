-- Revert the invite system: drop InvitedUser table + enums, restore User+Profile trigger.

-- 1) Drop the InvitedUser table (also drops its indexes and FK constraints)
DROP TABLE IF EXISTS platform."InvitedUser";

-- 2) Drop the enums introduced by the invite system
DROP TYPE IF EXISTS platform."InvitedUserStatus";
DROP TYPE IF EXISTS platform."TallyComputationStatus";

-- 3) Restore the User+Profile auto-creation trigger on auth.users.
--    Original definition from migration 20250205100104_add_profile_trigger.
--    generate_username() was never dropped and is still present.

CREATE OR REPLACE FUNCTION add_user_and_profile_to_platform()
RETURNS TRIGGER AS $$
BEGIN
    -- Exit early if NEW.id is null to prevent constraint violations
    IF NEW.id IS NULL THEN
        RAISE EXCEPTION 'Cannot create user/profile: id is null';
    END IF;

    /*
      1) Insert into platform."User"
         (If you already have such a row or want different columns, adjust below.)
    */
    INSERT INTO platform."User" (id, email, "updatedAt")
    VALUES (NEW.id, NEW.email, now());

    /*
      2) Insert into platform."Profile"
         Adjust columns/types depending on how your "Profile" schema is defined:
           - "links" might be text[], jsonb, or something else in your table.
           - "avatarUrl" and "description" can be defaulted as well.
    */
    INSERT INTO platform."Profile"
      ("id", "userId", name, username, description, links, "avatarUrl", "updatedAt")
    VALUES
      (
        NEW.id,
        NEW.id,
        COALESCE(split_part(NEW.email, '@', 1), 'user'),  -- handle null email
        platform.generate_username(),
        'I''m new here',
        '{}',                            -- empty array or empty JSON, depending on your column definition
        '',
        now()
      );

    RETURN NEW;
EXCEPTION
    WHEN OTHERS THEN
        -- Log the error details
        RAISE NOTICE 'Error in add_user_and_profile_to_platform: %', SQLERRM;
        -- Re-raise the error
        RAISE;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DO $$
BEGIN
    -- Check if the auth schema and users table exist
    IF EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'auth'
        AND table_name = 'users'
    ) THEN
        -- Drop the trigger if it exists
        DROP TRIGGER IF EXISTS user_added_to_platform ON auth.users;

        -- Create the trigger
        CREATE TRIGGER user_added_to_platform
        AFTER INSERT ON auth.users
        FOR EACH ROW EXECUTE FUNCTION add_user_and_profile_to_platform();
    END IF;
END $$;
