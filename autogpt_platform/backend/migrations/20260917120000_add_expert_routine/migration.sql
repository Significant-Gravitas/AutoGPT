-- CreateEnum
CREATE TYPE "ExpertRoutineSession" AS ENUM ('FRESH', 'HERE', 'THREAD');

-- CreateTable
CREATE TABLE "ExpertRoutine" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "expertId" TEXT NOT NULL,
    "key" TEXT,
    "title" TEXT NOT NULL,
    "prompt" TEXT NOT NULL,
    "crons" TEXT[],
    "asks" TEXT[],
    "sessionMode" "ExpertRoutineSession" NOT NULL DEFAULT 'THREAD',
    "sessionId" TEXT,
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

-- AddForeignKey
ALTER TABLE "ExpertRoutine" ADD CONSTRAINT "ExpertRoutine_expertId_fkey" FOREIGN KEY ("expertId") REFERENCES "Expert"("id") ON DELETE CASCADE ON UPDATE CASCADE;
