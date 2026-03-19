-- Revert the invite system: drop InvitedUser table + enums, restore User+Profile trigger.
-- All unqualified table/type names resolve via Prisma's configured schema context.
-- The trigger function sets its own search_path so it can find platform tables
-- when fired from the auth schema.

-- 1) Drop the InvitedUser table (also drops its indexes and FK constraints)
DROP TABLE IF EXISTS "InvitedUser";

-- 2) Drop the enums introduced by the invite system
DROP TYPE IF EXISTS "InvitedUserStatus";
DROP TYPE IF EXISTS "TallyComputationStatus";

-- 3) Restore the User+Profile auto-creation trigger on auth.users.
--    Original definition from migration 20250205100104_add_profile_trigger.
--    generate_username() was never dropped and is still present.
--    SET search_path ensures the function resolves tables in the correct schema
--    even when fired from the auth schema context.

CREATE OR REPLACE FUNCTION add_user_and_profile_to_platform()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = platform
AS $$
BEGIN
    IF NEW.id IS NULL THEN
        RAISE EXCEPTION 'Cannot create user/profile: id is null';
    END IF;

    INSERT INTO "User" (id, email, "updatedAt")
    VALUES (NEW.id, NEW.email, now());

    INSERT INTO "Profile"
      ("id", "userId", name, username, description, links, "avatarUrl", "updatedAt")
    VALUES
      (
        NEW.id,
        NEW.id,
        COALESCE(split_part(NEW.email, '@', 1), 'user'),
        generate_username(),
        'I''m new here',
        '{}',
        '',
        now()
      );

    RETURN NEW;
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Error in add_user_and_profile_to_platform: %', SQLERRM;
        RAISE;
END;
$$;

-- 4) Backfill: create User + Profile rows for any auth.users rows that were
--    created while the trigger was absent (during the invite-system window).
--    Uses explicit schema qualifiers since this DO block runs in Prisma's
--    schema context but needs to read from auth.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'auth' AND table_name = 'users'
    ) THEN
        INSERT INTO "User" (id, email, "updatedAt")
        SELECT au.id, au.email, now()
        FROM auth.users au
        LEFT JOIN "User" pu ON pu.id = au.id
        WHERE pu.id IS NULL
        ON CONFLICT (id) DO NOTHING;

        INSERT INTO "Profile"
          ("userId", name, username, description, links, "avatarUrl", "updatedAt")
        SELECT
          au.id,
          COALESCE(NULLIF(split_part(au.email, '@', 1), ''), 'user'),
          generate_username(),
          'I''m new here',
          '{}',
          '',
          now()
        FROM auth.users au
        LEFT JOIN "Profile" pp ON pp."userId" = au.id
        WHERE pp."userId" IS NULL
        ON CONFLICT ("userId") DO NOTHING;
    END IF;
END $$;

-- 5) Restore the trigger for future signups.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'auth'
        AND table_name = 'users'
    ) THEN
        DROP TRIGGER IF EXISTS user_added_to_platform ON auth.users;

        CREATE TRIGGER user_added_to_platform
        AFTER INSERT ON auth.users
        FOR EACH ROW EXECUTE FUNCTION add_user_and_profile_to_platform();
    END IF;
END $$;
