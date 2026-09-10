-- CreateTable
-- BotEvent is an append-only stream of discrete bot usage events. It NEVER
-- stores message content — only counts, timestamps, bounded enums and numeric
-- metrics. Powers message-volume, command-usage, error-rate and per-server
-- activity charts on the admin analytics page.
CREATE TABLE "BotEvent" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "platform" "PlatformType" NOT NULL,
    "eventType" TEXT NOT NULL,
    "serverId" TEXT,
    "channelType" TEXT,
    "commandName" TEXT,
    "errorKind" TEXT,
    "charCount" INTEGER,
    "durationMs" INTEGER,

    CONSTRAINT "BotEvent_pkey" PRIMARY KEY ("id")
);

-- CreateTable
-- BotGuild is the presence table: which servers the bot is physically in right
-- now (a superset of PlatformLink, since the bot can be in servers that never
-- ran /setup). leftAt IS NULL means currently joined; the joinedAt histogram
-- drives growth charts and sharding-threshold prediction.
CREATE TABLE "BotGuild" (
    "id" TEXT NOT NULL,
    "platform" "PlatformType" NOT NULL,
    "serverId" TEXT NOT NULL,
    "name" TEXT,
    "joinedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "leftAt" TIMESTAMP(3),
    "lastSeenAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "BotGuild_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "BotEvent_platform_eventType_createdAt_idx" ON "BotEvent"("platform", "eventType", "createdAt");

-- CreateIndex
CREATE INDEX "BotEvent_platform_serverId_createdAt_idx" ON "BotEvent"("platform", "serverId", "createdAt");

-- CreateIndex
CREATE INDEX "BotEvent_createdAt_idx" ON "BotEvent"("createdAt");

-- CreateIndex
CREATE UNIQUE INDEX "BotGuild_platform_serverId_key" ON "BotGuild"("platform", "serverId");

-- CreateIndex
CREATE INDEX "BotGuild_platform_leftAt_idx" ON "BotGuild"("platform", "leftAt");
