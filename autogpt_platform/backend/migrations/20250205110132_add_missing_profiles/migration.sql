DO $$
DECLARE
    -- Random username generation
    selected_adjective TEXT;
    selected_animal    TEXT;
    random_int         INT;
    generated_username TEXT;
    user_record       RECORD;
BEGIN
    -- Check if User and Profile tables exist
    IF EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'platform'
        AND table_name = 'User'
    ) AND EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'platform'
        AND table_name = 'Profile'
    ) THEN
        -- Loop through users without profiles
        FOR user_record IN 
            SELECT u.id, u.email
            FROM platform."User" u
            LEFT JOIN platform."Profile" p ON u.id = p."userId"
            WHERE p.id IS NULL
        LOOP
            -- Generate random username components
           FOR i IN 1..10 LOOP
        SELECT unnest
          INTO selected_adjective
          FROM (VALUES ('happy'), ('clever'), ('swift'), ('bright'), ('wise'), ('funny'), ('cool'), ('awesome'), ('amazing'), ('fantastic'), ('wonderful')) AS t(unnest)
          ORDER BY random()
          LIMIT 1;

        SELECT unnest
          INTO selected_animal
          FROM (VALUES ('fox'), ('wolf'), ('bear'), ('eagle'), ('owl'), ('tiger'), ('lion'), ('elephant'), ('giraffe'), ('zebra')) AS t(unnest)
          ORDER BY random()
          LIMIT 1;

        SELECT floor(random() * (99999 - 10000 + 1) + 10000)::int
          INTO random_int;

        generated_username := lower(selected_adjective || '-' || selected_animal || '_' || random_int);

        -- Check if username is already taken
        IF NOT EXISTS (
            SELECT 1 FROM platform."Profile" WHERE username = generated_username
        ) THEN
            -- Username is unique, exit the loop
            EXIT;
        END IF;

        -- If we've tried 10 times and still haven't found a unique username
        IF i = 10 THEN
            RAISE EXCEPTION 'Unable to generate unique username after 10 attempts';
        END IF;
        END LOOP;
            -- Create profile for user
            INSERT INTO platform."Profile"
              ("id", "userId", name, username, description, links, "avatarUrl", "updatedAt")
            VALUES
              (
                user_record.id,
                user_record.id,
                COALESCE(split_part(user_record.email, '@', 1), 'user'),
                generated_username,
                'I''m new here',
                '{}',
                '',
                now()
              );

        END LOOP;
    END IF;
END $$;