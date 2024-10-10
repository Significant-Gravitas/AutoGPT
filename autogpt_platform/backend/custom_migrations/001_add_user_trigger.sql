CREATE OR REPLACE FUNCTION add_user_to_platform() RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO platform."User" (id, email, "updatedAt")
    VALUES (NEW.id, NEW.email, now());
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER user_added_to_platform
AFTER INSERT ON auth.users
FOR EACH ROW
EXECUTE FUNCTION add_user_to_platform();