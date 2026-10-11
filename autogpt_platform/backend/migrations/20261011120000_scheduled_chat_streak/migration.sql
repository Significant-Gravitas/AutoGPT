-- One row per scheduled chat that is failing the same way, run after run
-- (SECRT-2879). A new table: nothing existing is rewritten.

-- CreateTable
CREATE TABLE "ScheduledChatStreak" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT NOT NULL,
    "scheduleKey" TEXT NOT NULL,
    "scheduleIds" TEXT[],
    "title" TEXT NOT NULL,
    "sessionId" TEXT,
    "failureKey" TEXT NOT NULL,
    "failureKind" TEXT NOT NULL,
    "provider" TEXT,
    "tool" TEXT,
    "failures" INTEGER NOT NULL DEFAULT 1,
    "wholeRunFailures" INTEGER NOT NULL DEFAULT 0,
    "notifiedAt" TIMESTAMP(3),
    "pausedAt" TIMESTAMP(3),

    CONSTRAINT "ScheduledChatStreak_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "ScheduledChatStreak_userId_pausedAt_idx" ON "ScheduledChatStreak"("userId", "pausedAt");

-- CreateIndex
CREATE UNIQUE INDEX "ScheduledChatStreak_userId_scheduleKey_key" ON "ScheduledChatStreak"("userId", "scheduleKey");

-- AddForeignKey
ALTER TABLE "ScheduledChatStreak" ADD CONSTRAINT "ScheduledChatStreak_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

