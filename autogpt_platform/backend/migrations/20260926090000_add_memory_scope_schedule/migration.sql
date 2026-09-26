-- The durable registry of memory scopes whose nightly dream and weekly
-- community-rebuild crons the scheduler runs: one row per scope (the account,
-- or one hired expert). Replaces the seven-day Redis registration markers as
-- the source of truth for "is this scope scheduled". A new, empty table: no
-- backfill runs here; `poetry run memory-schedule-backfill` fills it.

-- CreateEnum
CREATE TYPE "MemoryScopeScheduleState" AS ENUM ('ACTIVE', 'PAUSED', 'WIPED');

-- CreateTable
CREATE TABLE "MemoryScopeSchedule" (
    "scopeKey" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "expertId" TEXT,
    "timezone" TEXT NOT NULL,
    "state" "MemoryScopeScheduleState" NOT NULL DEFAULT 'ACTIVE',
    "communityJobId" TEXT,
    "nightlyJobId" TEXT,
    "lastNightlyRunAt" TIMESTAMP(3),
    "lastCommunityRunAt" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "MemoryScopeSchedule_pkey" PRIMARY KEY ("scopeKey"),
    -- The account's scope key is its user id; job ids and the scheduler's
    -- per-user wrappers rely on that. Prisma cannot express this, so the
    -- database holds it.
    CONSTRAINT "MemoryScopeSchedule_account_key_is_user_id" CHECK (("expertId" IS NOT NULL) OR ("scopeKey" = "userId"))
);

-- CreateIndex
CREATE UNIQUE INDEX "MemoryScopeSchedule_expertId_key" ON "MemoryScopeSchedule"("expertId");

-- CreateIndex
CREATE INDEX "MemoryScopeSchedule_userId_idx" ON "MemoryScopeSchedule"("userId");

-- CreateIndex
CREATE INDEX "MemoryScopeSchedule_state_idx" ON "MemoryScopeSchedule"("state");

-- AddForeignKey
ALTER TABLE "MemoryScopeSchedule" ADD CONSTRAINT "MemoryScopeSchedule_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "MemoryScopeSchedule" ADD CONSTRAINT "MemoryScopeSchedule_expertId_fkey" FOREIGN KEY ("expertId") REFERENCES "Expert"("id") ON DELETE CASCADE ON UPDATE CASCADE;
