DO $$
BEGIN
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
        INSERT INTO platform."Profile"
            ("id", "userId", name, username, description, links, "avatarUrl", "updatedAt")
        SELECT 
            u.id,
            u.id,
            COALESCE(split_part(u.email, '@', 1), 'user'),
            platform.generate_username(),
            'I''m new here',
            '{}',
            '',
            now()
        FROM platform."User" u
        LEFT JOIN platform."Profile" p ON u.id = p."userId"
        WHERE p.id IS NULL;
    END IF;
END $$;