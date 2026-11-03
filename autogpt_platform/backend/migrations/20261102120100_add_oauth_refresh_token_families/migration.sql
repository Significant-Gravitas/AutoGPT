-- Track refresh-token rotation families so replay of a consumed token
-- invalidates every descendant refresh and access token.
-- Existing access tokens cannot be linked reliably to an existing refresh token,
-- so they intentionally retain NULL and expire within their existing one-hour TTL.
-- This avoids a surprise mass revocation during rollout.
ALTER TABLE "OAuthAccessToken"
ADD COLUMN "refreshFamilyId" TEXT;

-- The database default is load-bearing: during a rolling deploy, servers still
-- on the previous release insert refresh tokens without a familyId, and every
-- existing row needs a family of its own. gen_random_uuid() is volatile, so
-- this rewrites OAuthRefreshToken under an ACCESS EXCLUSIVE lock, and the two
-- plain index builds below block writes while they run. Both tables were empty
-- in production on 2026-11-02, so that is instant. If they hold many rows when
-- this ships, stage it like 20260926160000/20260926160001: nullable column with
-- the default, backfill, NOT NULL, then indexes built CONCURRENTLY.
ALTER TABLE "OAuthRefreshToken"
ADD COLUMN "familyId" TEXT NOT NULL DEFAULT gen_random_uuid(),
ADD COLUMN "familyRevokedAt" TIMESTAMP(3);

CREATE INDEX "OAuthAccessToken_refreshFamilyId_idx"
ON "OAuthAccessToken"("refreshFamilyId");

CREATE INDEX "OAuthRefreshToken_familyId_idx"
ON "OAuthRefreshToken"("familyId");
