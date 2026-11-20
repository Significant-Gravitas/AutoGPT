-- CreateEnum
CREATE TYPE "AuthIPEventKind" AS ENUM ('SIGN_UP', 'SIGN_IN');

-- CreateEnum
CREATE TYPE "AuthAccessBlockKind" AS ENUM ('IP', 'EMAIL_DOMAIN');

-- CreateTable
CREATE TABLE "UserAuthIPEvent" (
    "id" TEXT NOT NULL DEFAULT gen_random_uuid(),
    "userId" TEXT NOT NULL,
    "kind" "AuthIPEventKind" NOT NULL,
    "ipAddress" TEXT NOT NULL,
    "userAgent" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "UserAuthIPEvent_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "UserAuthAccessBlock" (
    "id" TEXT NOT NULL DEFAULT gen_random_uuid(),
    "kind" "AuthAccessBlockKind" NOT NULL,
    "value" TEXT NOT NULL,
    "reason" TEXT NOT NULL,
    "createdById" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "expiresAt" TIMESTAMP(3),

    CONSTRAINT "UserAuthAccessBlock_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "UserAuthIPEvent_userId_createdAt_idx" ON "UserAuthIPEvent"("userId", "createdAt");

-- CreateIndex
CREATE INDEX "UserAuthIPEvent_ipAddress_idx" ON "UserAuthIPEvent"("ipAddress");

-- CreateIndex
CREATE INDEX "UserAuthIPEvent_createdAt_idx" ON "UserAuthIPEvent"("createdAt");

-- CreateIndex
CREATE UNIQUE INDEX "UserAuthAccessBlock_kind_value_key" ON "UserAuthAccessBlock"("kind", "value");

-- AddForeignKey
ALTER TABLE "UserAuthIPEvent" ADD CONSTRAINT "UserAuthIPEvent_userId_fkey" FOREIGN KEY ("userId") REFERENCES "UserAuthIdentity"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- Seed the log with the IPs of the sessions that exist now, so the admin page
-- has something to correlate before the first new sign-in. Impersonation
-- sessions carry the admin's IP, not the user's.
INSERT INTO "UserAuthIPEvent" ("userId", "kind", "ipAddress", "userAgent", "createdAt")
SELECT "userId", 'SIGN_IN', "ipAddress", NULLIF("userAgent", ''), "createdAt"
FROM "UserAuthSession"
WHERE "ipAddress" IS NOT NULL
  AND "ipAddress" <> ''
  AND "impersonatedBy" IS NULL;

-- The frontend's Better Auth hooks call these (lib/auth/access-blocks.ts and
-- ip-log.ts), so the rules they run are tested here, against a database.
CREATE FUNCTION "is_auth_access_blocked"(blocked_ip TEXT, email_domains TEXT[])
RETURNS BOOLEAN
LANGUAGE sql STABLE
SET search_path FROM CURRENT
AS $$
  SELECT EXISTS (
    SELECT 1 FROM "UserAuthAccessBlock"
    WHERE ("expiresAt" IS NULL OR "expiresAt" > now())
      AND (("kind" = 'IP' AND "value" = blocked_ip)
        OR ("kind" = 'EMAIL_DOMAIN' AND "value" = ANY(email_domains)))
  )
$$;

CREATE FUNCTION "record_auth_ip_event"(
  event_user_id TEXT, event_kind TEXT, event_ip TEXT, event_user_agent TEXT
)
RETURNS VOID
LANGUAGE sql
SET search_path FROM CURRENT
AS $$
  INSERT INTO "UserAuthIPEvent" ("userId", "kind", "ipAddress", "userAgent")
  VALUES (event_user_id, event_kind::"AuthIPEventKind", event_ip, NULLIF(event_user_agent, ''))
$$;
