-- CreateEnum
CREATE TYPE "ExpertRoutineSession" AS ENUM ('FRESH', 'PINNED', 'THREAD');

-- CreateEnum
CREATE TYPE "ExpertRoutineSource" AS ENUM ('TEMPLATE', 'OWNER');

-- CreateTable
CREATE TABLE "ExpertRoutine" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "expertId" TEXT,
    "userId" TEXT,
    "key" TEXT,
    "title" TEXT NOT NULL,
    "prompt" TEXT NOT NULL,
    "crons" TEXT[],
    "asks" TEXT[],
    "sessionMode" "ExpertRoutineSession" NOT NULL DEFAULT 'THREAD',
    "sessionId" TEXT,
    "source" "ExpertRoutineSource" NOT NULL DEFAULT 'TEMPLATE',
    "runAt" TIMESTAMP(3),
    "firedAt" TIMESTAMP(3),
    "scheduleIds" TEXT[],
    "enabledAt" TIMESTAMP(3),
    "customizedAt" TIMESTAMP(3),
    "grantsCredentials" BOOLEAN NOT NULL DEFAULT false,
    "pausedByExpertArchive" BOOLEAN NOT NULL DEFAULT false,

    CONSTRAINT "ExpertRoutine_pkey" PRIMARY KEY ("id"),
    -- Exactly one owner. Prisma cannot express this, so both columns are
    -- nullable in the schema and every reader has to trust the invariant;
    -- stated here it is the database's job. A row with both owners would be
    -- reachable from two scopes at once, and a row with neither would be
    -- reachable from none while still firing.
    CONSTRAINT "ExpertRoutine_one_owner" CHECK (("expertId" IS NULL) <> ("userId" IS NULL))
);

-- CreateIndex
CREATE UNIQUE INDEX "ExpertRoutine_expertId_key_key" ON "ExpertRoutine"("expertId", "key");

-- CreateIndex
CREATE INDEX "ExpertRoutine_expertId_createdAt_id_idx" ON "ExpertRoutine"("expertId", "createdAt", "id");

-- CreateIndex
CREATE INDEX "ExpertRoutine_userId_createdAt_id_idx" ON "ExpertRoutine"("userId", "createdAt", "id");

-- AddForeignKey
ALTER TABLE "ExpertRoutine" ADD CONSTRAINT "ExpertRoutine_expertId_fkey" FOREIGN KEY ("expertId") REFERENCES "Expert"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ExpertRoutine" ADD CONSTRAINT "ExpertRoutine_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
