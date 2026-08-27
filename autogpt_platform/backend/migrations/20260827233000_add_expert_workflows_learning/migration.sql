ALTER TABLE "ExpertWorkflow"
ADD COLUMN "purpose" TEXT,
ADD COLUMN "expectedInputs" TEXT,
ADD COLUMN "expectedOutputs" TEXT,
ADD COLUMN "cadence" TEXT;

CREATE UNIQUE INDEX "ExpertWorkflow_expertId_libraryAgentId_key"
ON "ExpertWorkflow"("expertId", "libraryAgentId");

CREATE TYPE "ExpertLearnedNoteStatus" AS ENUM ('ACTIVE', 'ARCHIVED');

CREATE TABLE "ExpertLearnedNote" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT NOT NULL,
    "expertId" TEXT NOT NULL,
    "text" TEXT NOT NULL,
    "dedupeKey" TEXT NOT NULL,
    "learnedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "sourceSessionId" TEXT,
    "sourceRuleId" TEXT,
    "status" "ExpertLearnedNoteStatus" NOT NULL DEFAULT 'ACTIVE',

    CONSTRAINT "ExpertLearnedNote_pkey" PRIMARY KEY ("id")
);

CREATE INDEX "ExpertLearnedNote_userId_expertId_status_idx"
ON "ExpertLearnedNote"("userId", "expertId", "status");

CREATE INDEX "ExpertLearnedNote_userId_sourceRuleId_idx"
ON "ExpertLearnedNote"("userId", "sourceRuleId");

CREATE UNIQUE INDEX "ExpertLearnedNote_userId_expertId_dedupeKey_key"
ON "ExpertLearnedNote"("userId", "expertId", "dedupeKey");

ALTER TABLE "ExpertLearnedNote"
ADD CONSTRAINT "ExpertLearnedNote_userId_fkey"
FOREIGN KEY ("userId") REFERENCES "User"("id")
ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "ExpertLearnedNote"
ADD CONSTRAINT "ExpertLearnedNote_expertId_fkey"
FOREIGN KEY ("expertId") REFERENCES "Expert"("id")
ON DELETE CASCADE ON UPDATE CASCADE;
