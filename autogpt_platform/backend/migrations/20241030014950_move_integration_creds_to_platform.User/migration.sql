-- Migrate integration credentials from auth.user.raw_user_meta_data to platform.User.metadata
DO $$
BEGIN
  IF EXISTS (
      SELECT 1
      FROM information_schema.tables
      WHERE table_schema = 'auth'
      AND table_name = 'users'
  ) THEN
    -- First update User metadata for users that have integration_credentials
    WITH users_with_creds AS (
      SELECT
        id,
        raw_user_meta_data->'integration_credentials' as integration_credentials,
        raw_user_meta_data
      FROM auth.users
      WHERE raw_user_meta_data ? 'integration_credentials'
    )
    UPDATE "User" u
    SET metadata = COALESCE(
      CASE
        -- If User.metadata already has .integration_credentials, leave it
        WHEN u.metadata ? 'integration_credentials' THEN u.metadata
        -- If User.metadata exists but has no .integration_credentials, add it
        WHEN u.metadata IS NOT NULL AND u.metadata::text != '' THEN
          (u.metadata || jsonb_build_object('integration_credentials', uwc.integration_credentials))
        -- If User.metadata is NULL, set it
        ELSE jsonb_build_object('integration_credentials', uwc.integration_credentials)
      END,
      '{}'::jsonb
    )
    FROM users_with_creds uwc
    WHERE u.id = uwc.id::text;

  -- Finally remove integration_credentials from auth.users
  UPDATE auth.users
  SET raw_user_meta_data = raw_user_meta_data - 'integration_credentials'
  WHERE raw_user_meta_data ? 'integration_credentials';
  END IF;
END $$;
