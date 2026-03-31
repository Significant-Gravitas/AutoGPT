-- CreateEnum
CREATE TYPE "PlatformType" AS ENUM ('DISCORD', 'TELEGRAM', 'SLACK', 'TEAMS', 'WHATSAPP', 'GITHUB', 'LINEAR');

-- CreateTable
CREATE TABLE "PlatformLink" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "platform" "PlatformType" NOT NULL,
    "platformUserId" TEXT NOT NULL,
    "platformUsername" TEXT,
    "linkedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "PlatformLink_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "PlatformLinkToken" (
    "id" TEXT NOT NULL,
    "token" TEXT NOT NULL,
    "platform" "PlatformType" NOT NULL,
    "platformUserId" TEXT NOT NULL,
    "platformUsername" TEXT,
    "channelId" TEXT,
    "expiresAt" TIMESTAMP(3) NOT NULL,
    "usedAt" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "PlatformLinkToken_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "PlatformLink_platform_platformUserId_key" ON "PlatformLink"("platform", "platformUserId");

-- CreateIndex
CREATE INDEX "PlatformLink_userId_idx" ON "PlatformLink"("userId");

-- CreateIndex
CREATE UNIQUE INDEX "PlatformLinkToken_token_key" ON "PlatformLinkToken"("token");

-- CreateIndex
CREATE INDEX "PlatformLinkToken_token_idx" ON "PlatformLinkToken"("token");

-- CreateIndex
CREATE INDEX "PlatformLinkToken_expiresAt_idx" ON "PlatformLinkToken"("expiresAt");

-- AddForeignKey
ALTER TABLE "PlatformLink" ADD CONSTRAINT "PlatformLink_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
