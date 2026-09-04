-- Better Auth additional field backing the onboarding "What should I call
-- you?" answer (user_metadata.preferred_name in the GoTrue world). Must run
-- before 20260716120000_copy_supabase_users_to_better_auth, which copies the
-- GoTrue metadata value into it.
ALTER TABLE "UserAuthIdentity" ADD COLUMN "preferredName" TEXT;
