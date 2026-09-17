-- CreateTable
CREATE TABLE "BotInstall" (
    "id" TEXT NOT NULL,
    "platform" "PlatformType" NOT NULL,
    "platformServerId" TEXT NOT NULL,
    "credentials" TEXT NOT NULL,
    "botUserId" TEXT,
    "appId" TEXT,
    "name" TEXT,
    "installedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3),
    "revokedAt" TIMESTAMP(3),

    CONSTRAINT "BotInstall_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "BotInstall_platform_platformServerId_key" ON "BotInstall"("platform", "platformServerId");

-- CreateIndex
CREATE INDEX "BotInstall_platform_revokedAt_idx" ON "BotInstall"("platform", "revokedAt");
