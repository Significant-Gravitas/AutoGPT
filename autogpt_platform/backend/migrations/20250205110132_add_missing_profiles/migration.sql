DO $$
DECLARE
    -- Random username generation
    selected_adjective TEXT;
    selected_animal    TEXT;
    random_int         INT;
    generated_username TEXT;
    user_record       RECORD;
BEGIN
    -- Loop through users without profiles
    FOR user_record IN 
        SELECT u.id, u.email
        FROM platform."User" u
        LEFT JOIN platform."Profile" p ON u.id = p."userId"
        WHERE p.id IS NULL
    LOOP
        -- Generate random username components
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
END $$;
