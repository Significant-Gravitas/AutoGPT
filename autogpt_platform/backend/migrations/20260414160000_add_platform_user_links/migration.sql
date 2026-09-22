-- CreateEnum
-- Server links (group chats / guilds) and user links (personal DMs) are
-- fully independent — a user who owns a linked server still has to link
-- their DMs separately.
CREATE TYPE "PlatformLinkType" AS ENUM ('SERVER', 'USER');

-- CreateTable
-- PlatformUserLink maps an individual platform user identity to an AutoGPT
-- account for 1:1 DMs with the bot. Independent from PlatformLink.
CREATE TABLE "PlatformUserLink" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "platform" "PlatformType" NOT NULL,
    "platformUserId" TEXT NOT NULL,
    "platformUsername" TEXT,
    "linkedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "PlatformUserLink_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "PlatformUserLink_platform_platformUserId_key" ON "PlatformUserLink"("platform", "platformUserId");

-- CreateIndex
CREATE INDEX "PlatformUserLink_userId_idx" ON "PlatformUserLink"("userId");

-- AddForeignKey
ALTER TABLE "PlatformUserLink" ADD CONSTRAINT "PlatformUserLink_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AlterTable: PlatformLinkToken now supports SERVER or USER tokens.
-- Existing rows are all SERVER (default matches the column default).
ALTER TABLE "PlatformLinkToken"
    ADD COLUMN "linkType" "PlatformLinkType" NOT NULL DEFAULT 'SERVER',
    ALTER COLUMN "platformServerId" DROP NOT NULL;

-- CreateIndex
CREATE INDEX "PlatformLinkToken_platform_platformUserId_idx" ON "PlatformLinkToken"("platform", "platformUserId");
