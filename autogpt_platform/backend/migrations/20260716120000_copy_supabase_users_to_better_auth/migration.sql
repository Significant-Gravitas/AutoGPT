-- One-time copy of Supabase GoTrue users into the Better Auth tables:
--   auth.users                    -> "UserAuthIdentity"
--   email/password credentials    -> "UserAuthAccount" (providerId = 'credential';
--                                   bcrypt hashes carry over because Better
--                                   Auth is configured to verify with bcrypt)
--   auth.identities               -> "UserAuthAccount" (google / github / discord)
--
-- Runs in the normal `prisma migrate deploy` pipeline step, which the deploy
-- workflows order before any pod rollout. All data movement is server-side
-- INSERT..SELECT, so the statement count is tiny and the runner's network
-- location is irrelevant.
--
-- On databases without a Supabase auth schema (fresh installs, CI, local
-- dev) the guard makes this a no-op. Every insert is also individually
-- guarded (ON CONFLICT / NOT EXISTS) so the migration is safe on databases
-- where some users were already copied — e.g. preview environments where
-- scripts/migrate-supabase-auth.ts ran manually. That same script remains
-- the post-cutover straggler sweep for users who sign up via GoTrue between
-- this migration running and the frontend switching to Better Auth.

DO $$
DECLARE
  users_copied bigint := 0;
  credential_accounts_copied bigint := 0;
  oauth_accounts_copied bigint := 0;
BEGIN
  IF to_regclass('auth.users') IS NULL THEN
    RAISE NOTICE 'auth.users does not exist - nothing to migrate';
    RETURN;
  END IF;

  -- 1) auth.users -> "UserAuthIdentity". Skip deleted users and rows without an email.
  --    The NOT EXISTS email guard skips users whose email is already taken
  --    by a different Better Auth user instead of failing the whole
  --    migration on the unique(email) index.
  --
  --    DISTINCT ON (u.email) de-dups WITHIN this batch: two GoTrue rows can
  --    share an email (e.g. an SSO identity + a password identity — GoTrue's
  --    partial unique index permits it). Without it, both rows pass the
  --    NOT EXISTS guard (neither is in "UserAuthIdentity" yet) and the second insert trips
  --    UserAuthIdentity_email_key, aborting the ENTIRE INSERT and failing the deploy. The
  --    ORDER BY makes the surviving row deterministic: prefer a confirmed
  --    account, then the oldest. The dropped duplicate's credential/oauth rows
  --    are naturally skipped below by the "id must exist in UserAuthIdentity" guards.
  INSERT INTO "UserAuthIdentity"
    (id, name, email, "emailVerified", role, banned, "preferredName",
     "createdAt", "updatedAt")
  SELECT DISTINCT ON (u.email)
    u.id::text,
    COALESCE(
      u.raw_user_meta_data->>'name',
      u.raw_user_meta_data->>'full_name',
      split_part(u.email, '@', 1)
    ),
    u.email,
    (u.email_confirmed_at IS NOT NULL),
    CASE
      WHEN COALESCE(u.is_super_admin, false) OR u.role = 'admin'
      THEN 'admin' ELSE 'user'
    END,
    (u.banned_until IS NOT NULL AND u.banned_until > now()),
    u.raw_user_meta_data->>'preferred_name',
    COALESCE(u.created_at, now()),
    COALESCE(u.updated_at, now())
  FROM auth.users u
  WHERE u.email IS NOT NULL
    AND u.deleted_at IS NULL
    AND NOT EXISTS (
      SELECT 1 FROM "UserAuthIdentity" pu
      WHERE pu.email = u.email AND pu.id <> u.id::text
    )
  ORDER BY
    u.email,
    (u.email_confirmed_at IS NOT NULL) DESC,
    COALESCE(u.created_at, now()) ASC
  ON CONFLICT (id) DO NOTHING;
  GET DIAGNOSTICS users_copied = ROW_COUNT;

  -- 2) Email/password credentials -> "UserAuthAccount".
  INSERT INTO "UserAuthAccount"
    (id, "accountId", "providerId", "userId", password, "createdAt", "updatedAt")
  SELECT
    gen_random_uuid()::text,
    u.id::text,
    'credential',
    u.id::text,
    u.encrypted_password,
    COALESCE(u.created_at, now()),
    COALESCE(u.updated_at, now())
  FROM auth.users u
  WHERE u.encrypted_password IS NOT NULL
    AND length(u.encrypted_password) > 0
    AND EXISTS (
      SELECT 1 FROM "UserAuthIdentity" pu WHERE pu.id = u.id::text
    )
    AND NOT EXISTS (
      SELECT 1 FROM "UserAuthAccount" a
      WHERE a."userId" = u.id::text AND a."providerId" = 'credential'
    );
  GET DIAGNOSTICS credential_accounts_copied = ROW_COUNT;

  -- 3) OAuth identities -> account. Provider 'email' is the GoTrue-internal
  --    credential identity and is skipped (handled above); only providers
  --    Better Auth is configured for migrate. GoTrue added
  --    identities.provider_id (the provider's user id) in newer versions;
  --    older schemas only carry it inside identity_data->>'sub'. The branch
  --    referencing provider_id only resolves when that column exists.
  IF to_regclass('auth.identities') IS NOT NULL THEN
    IF EXISTS (
      SELECT 1 FROM information_schema.columns
      WHERE table_schema = 'auth'
        AND table_name = 'identities'
        AND column_name = 'provider_id'
    ) THEN
      INSERT INTO "UserAuthAccount"
        (id, "accountId", "providerId", "userId", "createdAt", "updatedAt")
      SELECT
        gen_random_uuid()::text,
        COALESCE(i.provider_id::text, i.identity_data->>'sub', i.user_id::text),
        i.provider,
        i.user_id::text,
        COALESCE(i.created_at, now()),
        COALESCE(i.updated_at, now())
      FROM auth.identities i
      WHERE i.provider IN ('google', 'github', 'discord')
        AND EXISTS (
          SELECT 1 FROM "UserAuthIdentity" pu WHERE pu.id = i.user_id::text
        )
        AND NOT EXISTS (
          SELECT 1 FROM "UserAuthAccount" a
          WHERE a."userId" = i.user_id::text AND a."providerId" = i.provider
        );
    ELSE
      INSERT INTO "UserAuthAccount"
        (id, "accountId", "providerId", "userId", "createdAt", "updatedAt")
      SELECT
        gen_random_uuid()::text,
        COALESCE(i.identity_data->>'sub', i.user_id::text),
        i.provider,
        i.user_id::text,
        COALESCE(i.created_at, now()),
        COALESCE(i.updated_at, now())
      FROM auth.identities i
      WHERE i.provider IN ('google', 'github', 'discord')
        AND EXISTS (
          SELECT 1 FROM "UserAuthIdentity" pu WHERE pu.id = i.user_id::text
        )
        AND NOT EXISTS (
          SELECT 1 FROM "UserAuthAccount" a
          WHERE a."userId" = i.user_id::text AND a."providerId" = i.provider
        );
    END IF;
    GET DIAGNOSTICS oauth_accounts_copied = ROW_COUNT;
  END IF;

  RAISE NOTICE 'supabase -> better auth copy: % users, % credential accounts, % oauth accounts',
    users_copied, credential_accounts_copied, oauth_accounts_copied;
END $$;
