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

    CONSTRAINT "ExpertRoutine_pkey" PRIMARY KEY ("id")
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
