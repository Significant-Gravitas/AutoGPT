-- CreateEnum
CREATE TYPE "ExpertLearnedNoteStatus" AS ENUM ('ACTIVE', 'ARCHIVED');

-- CreateTable
CREATE TABLE "ExpertLearnedNote" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT NOT NULL,
    "expertId" TEXT,
    "text" TEXT NOT NULL,
    "learnedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "sourceSessionId" TEXT,
    "sourceRuleId" TEXT,
    "status" "ExpertLearnedNoteStatus" NOT NULL DEFAULT 'ACTIVE',

    CONSTRAINT "ExpertLearnedNote_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "ExpertLearnedNote_userId_expertId_status_idx" ON "ExpertLearnedNote"("userId", "expertId", "status");

-- CreateIndex
CREATE INDEX "ExpertLearnedNote_userId_sourceRuleId_idx" ON "ExpertLearnedNote"("userId", "sourceRuleId");

-- AddForeignKey
ALTER TABLE "ExpertLearnedNote" ADD CONSTRAINT "ExpertLearnedNote_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ExpertLearnedNote" ADD CONSTRAINT "ExpertLearnedNote_expertId_fkey" FOREIGN KEY ("expertId") REFERENCES "Expert"("id") ON DELETE CASCADE ON UPDATE CASCADE;
