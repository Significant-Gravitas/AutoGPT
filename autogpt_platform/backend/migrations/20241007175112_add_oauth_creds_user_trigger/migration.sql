--CreateFunction
CREATE OR REPLACE FUNCTION add_user_to_platform() RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO platform."User" (id, email, "updatedAt")
    VALUES (NEW.id, NEW.email, now());
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- This CREATE commands is for dev-only, where full-fledged Supabase can be missing.
-- Create the 'auth' schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS auth;
CREATE TABLE IF NOT EXISTS auth.users (id UUID PRIMARY KEY, email TEXT);

-- DropTigger
DROP TRIGGER IF EXISTS user_added_to_platform ON auth.users;

--CreateTrigger
CREATE TRIGGER user_added_to_platform AFTER INSERT ON auth.users
FOR EACH ROW EXECUTE FUNCTION add_user_to_platform();
