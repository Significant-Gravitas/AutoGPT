CREATE OR REPLACE FUNCTION add_user_and_profile_to_platform()
RETURNS TRIGGER AS $$
DECLARE
    -- Random username generation
    selected_adjective TEXT;
    selected_animal    TEXT;
    random_int         INT;
    generated_username TEXT;
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
      2) Generate a random username
         -- This replicates your Python logic of choosing a random adjective,
            a random animal, and a 4-digit number between 1000-9999.
    */
    SELECT unnest
      INTO selected_adjective
      FROM (VALUES ('happy'), ('clever'), ('swift'), ('bright'), ('wise')) AS t(unnest)
      ORDER BY random()
      LIMIT 1;

    SELECT unnest
      INTO selected_animal
      FROM (VALUES ('fox'), ('wolf'), ('bear'), ('eagle'), ('owl')) AS t(unnest)
      ORDER BY random()
      LIMIT 1;

    SELECT floor(random() * (9999 - 1000 + 1) + 1000)::int
      INTO random_int;

    generated_username := lower(selected_adjective || '-' || selected_animal || '_' || random_int);

    /*
      3) Insert into platform."Profile"
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
        generated_username,
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


-- Drop the old trigger if it exists
DROP TRIGGER IF EXISTS user_added_to_platform ON auth.users;

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

