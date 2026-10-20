-- CreateEnum
CREATE TYPE "AuthIpEventKind" AS ENUM ('SIGN_UP', 'SIGN_IN');

-- CreateEnum
CREATE TYPE "AuthAccessBlockKind" AS ENUM ('IP', 'EMAIL_DOMAIN');

-- CreateTable
CREATE TABLE "UserAuthIpEvent" (
    "id" TEXT NOT NULL DEFAULT gen_random_uuid(),
    "userId" TEXT NOT NULL,
    "kind" "AuthIpEventKind" NOT NULL,
    "ipAddress" TEXT NOT NULL,
    "userAgent" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "UserAuthIpEvent_pkey" PRIMARY KEY ("id")
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
CREATE INDEX "UserAuthIpEvent_userId_createdAt_idx" ON "UserAuthIpEvent"("userId", "createdAt");

-- CreateIndex
CREATE INDEX "UserAuthIpEvent_ipAddress_idx" ON "UserAuthIpEvent"("ipAddress");

-- CreateIndex
CREATE INDEX "UserAuthIpEvent_createdAt_idx" ON "UserAuthIpEvent"("createdAt");

-- CreateIndex
CREATE UNIQUE INDEX "UserAuthAccessBlock_kind_value_key" ON "UserAuthAccessBlock"("kind", "value");

-- AddForeignKey
ALTER TABLE "UserAuthIpEvent" ADD CONSTRAINT "UserAuthIpEvent_userId_fkey" FOREIGN KEY ("userId") REFERENCES "UserAuthIdentity"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- Seed the log with the IPs of the sessions that exist now, so the admin page
-- has something to correlate before the first new sign-in. Impersonation
-- sessions carry the admin's IP, not the user's.
INSERT INTO "UserAuthIpEvent" ("userId", "kind", "ipAddress", "userAgent", "createdAt")
SELECT "userId", 'SIGN_IN', "ipAddress", NULLIF("userAgent", ''), "createdAt"
FROM "UserAuthSession"
WHERE "ipAddress" IS NOT NULL
  AND "ipAddress" <> ''
  AND "impersonatedBy" IS NULL;
