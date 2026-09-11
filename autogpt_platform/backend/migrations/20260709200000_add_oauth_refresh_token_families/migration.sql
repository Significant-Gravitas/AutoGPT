-- Track refresh-token rotation families so replay of a consumed token
-- invalidates every descendant refresh and access token.
-- Existing access tokens cannot be linked reliably to an existing refresh token,
-- so they intentionally retain NULL and expire within their existing one-hour TTL.
-- This avoids a surprise mass revocation during rollout.
ALTER TABLE "OAuthAccessToken"
ADD COLUMN "refreshFamilyId" TEXT;

ALTER TABLE "OAuthRefreshToken"
ADD COLUMN "familyId" TEXT NOT NULL DEFAULT gen_random_uuid(),
ADD COLUMN "familyRevokedAt" TIMESTAMP(3);

CREATE INDEX "OAuthAccessToken_refreshFamilyId_idx"
ON "OAuthAccessToken"("refreshFamilyId");

CREATE INDEX "OAuthRefreshToken_familyId_idx"
ON "OAuthRefreshToken"("familyId");
