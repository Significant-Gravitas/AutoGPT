-- AlterTable
ALTER TABLE "AgentPreset" ADD COLUMN "expertId" TEXT;

-- AlterTable
ALTER TABLE "Expert" ADD COLUMN "weeklyBudget" INTEGER,
ADD COLUMN "schedulesPausedAt" TIMESTAMP(3);

-- CreateTable
CREATE TABLE "ExpertPauseEvent" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "expertId" TEXT NOT NULL,
    "reason" TEXT NOT NULL,
    "clearedAt" TIMESTAMP(3),

    CONSTRAINT "ExpertPauseEvent_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "ExpertPauseEvent_expertId_createdAt_idx" ON "ExpertPauseEvent"("expertId", "createdAt");

-- CreateIndex
-- AgentPreset is modest-sized: the plain build's SHARE lock window is
-- short, so the index ships inline with the column it indexes. (No
-- out-of-band escape hatch — expertId only exists once this migration
-- runs; IF NOT EXISTS just keeps a rerun idempotent.)
CREATE INDEX IF NOT EXISTS "AgentPreset_expertId_idx" ON "AgentPreset"("expertId");

-- AddForeignKey
ALTER TABLE "AgentPreset" ADD CONSTRAINT "AgentPreset_expertId_fkey" FOREIGN KEY ("expertId") REFERENCES "Expert"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ExpertPauseEvent" ADD CONSTRAINT "ExpertPauseEvent_expertId_fkey" FOREIGN KEY ("expertId") REFERENCES "Expert"("id") ON DELETE CASCADE ON UPDATE CASCADE;
