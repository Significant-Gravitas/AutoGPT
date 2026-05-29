--CreateFunction
CREATE OR REPLACE FUNCTION add_user_to_platform() RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO platform."User" (id, email, "updatedAt")
    VALUES (NEW.id, NEW.email, now());
    RETURN NEW;
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
        FOR EACH ROW EXECUTE FUNCTION add_user_to_platform();
    END IF;
END $$;
