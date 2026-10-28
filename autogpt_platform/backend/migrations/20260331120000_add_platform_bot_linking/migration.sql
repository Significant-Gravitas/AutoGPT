-- CreateEnum
CREATE TYPE "PlatformType" AS ENUM ('DISCORD', 'TELEGRAM', 'SLACK', 'TEAMS', 'WHATSAPP', 'GITHUB', 'LINEAR');

-- CreateTable
-- PlatformLink maps a platform server (Discord guild, Telegram group, etc.) to an AutoGPT
-- owner account. The first user to authenticate becomes the owner — all usage from that
-- server is billed to their account. Each user within the server gets their own CoPilot
-- session, all visible in the owner's AutoGPT account.
CREATE TABLE "PlatformLink" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "platform" "PlatformType" NOT NULL,
    "platformServerId" TEXT NOT NULL,
    "ownerPlatformUserId" TEXT NOT NULL,
    "serverName" TEXT,
    "linkedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "PlatformLink_pkey" PRIMARY KEY ("id")
);

-- CreateTable
-- PlatformLinkToken is a one-time token for the server linking flow.
CREATE TABLE "PlatformLinkToken" (
    "id" TEXT NOT NULL,
    "token" TEXT NOT NULL,
    "platform" "PlatformType" NOT NULL,
    "platformServerId" TEXT NOT NULL,
    "platformUserId" TEXT NOT NULL,
    "platformUsername" TEXT,
    "serverName" TEXT,
    "channelId" TEXT,
    "expiresAt" TIMESTAMP(3) NOT NULL,
    "usedAt" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "PlatformLinkToken_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "PlatformLink_platform_platformServerId_key" ON "PlatformLink"("platform", "platformServerId");

-- CreateIndex
CREATE INDEX "PlatformLink_userId_idx" ON "PlatformLink"("userId");

-- CreateIndex
CREATE UNIQUE INDEX "PlatformLinkToken_token_key" ON "PlatformLinkToken"("token");

-- CreateIndex
CREATE INDEX "PlatformLinkToken_platform_platformServerId_idx" ON "PlatformLinkToken"("platform", "platformServerId");

-- CreateIndex
CREATE INDEX "PlatformLinkToken_expiresAt_idx" ON "PlatformLinkToken"("expiresAt");

-- AddForeignKey
ALTER TABLE "PlatformLink" ADD CONSTRAINT "PlatformLink_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
