CREATE TYPE "ExpertWorkItemStatus" AS ENUM (
  'QUEUED',
  'RUNNING',
  'DELIVERED',
  'PARTIAL',
  'BLOCKED_MANAGER',
  'FAILED'
);

CREATE TYPE "ExpertWorkConfidence" AS ENUM (
  'VERIFIED',
  'LIKELY',
  'UNKNOWN',
  'DISQUALIFIED'
);

CREATE TABLE "ExpertWorkItem" (
  "id" TEXT NOT NULL,
  "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "ownerUserId" TEXT NOT NULL,
  "expertId" TEXT NOT NULL,
  "managerSessionId" TEXT NOT NULL,
  "delegatedSessionId" TEXT NOT NULL,
  "projectPhase" TEXT NOT NULL DEFAULT '',
  "taskTitle" TEXT NOT NULL,
  "expectedDeliverable" TEXT NOT NULL,
  "deliverableMode" TEXT NOT NULL DEFAULT 'message',
  "successCriteria" JSONB NOT NULL DEFAULT '[]',
  "dependencies" TEXT[] DEFAULT ARRAY[]::TEXT[],
  "sourceArtifacts" JSONB NOT NULL DEFAULT '[]',
  "constraints" TEXT[] DEFAULT ARRAY[]::TEXT[],
  "approvalBoundaries" TEXT[] DEFAULT ARRAY[]::TEXT[],
  "estimateMinutes" INTEGER,
  "progress" INTEGER NOT NULL DEFAULT 0,
  "status" "ExpertWorkItemStatus" NOT NULL DEFAULT 'QUEUED',
  "result" TEXT,
  "blocker" TEXT,
  "confidence" "ExpertWorkConfidence" NOT NULL DEFAULT 'UNKNOWN',
  "artifacts" JSONB NOT NULL DEFAULT '[]',
  "startedAt" TIMESTAMP(3),
  "completedAt" TIMESTAMP(3),
  "managerWaitExpiresAt" TIMESTAMP(3),
  "parentWokenAt" TIMESTAMP(3),
  CONSTRAINT "ExpertWorkItem_pkey" PRIMARY KEY ("id")
);

CREATE INDEX "ExpertWorkItem_ownerUserId_createdAt_idx"
  ON "ExpertWorkItem"("ownerUserId", "createdAt");
CREATE INDEX "ExpertWorkItem_expertId_status_updatedAt_idx"
  ON "ExpertWorkItem"("expertId", "status", "updatedAt");
CREATE INDEX "ExpertWorkItem_managerSessionId_updatedAt_idx"
  ON "ExpertWorkItem"("managerSessionId", "updatedAt");
CREATE INDEX "ExpertWorkItem_delegatedSessionId_updatedAt_idx"
  ON "ExpertWorkItem"("delegatedSessionId", "updatedAt");

ALTER TABLE "ExpertWorkItem"
  ADD CONSTRAINT "ExpertWorkItem_ownerUserId_fkey"
  FOREIGN KEY ("ownerUserId") REFERENCES "User"("id")
  ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "ExpertWorkItem"
  ADD CONSTRAINT "ExpertWorkItem_expertId_fkey"
  FOREIGN KEY ("expertId") REFERENCES "Expert"("id")
  ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "ExpertWorkItem"
  ADD CONSTRAINT "ExpertWorkItem_managerSessionId_fkey"
  FOREIGN KEY ("managerSessionId") REFERENCES "ChatSession"("id")
  ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "ExpertWorkItem"
  ADD CONSTRAINT "ExpertWorkItem_delegatedSessionId_fkey"
  FOREIGN KEY ("delegatedSessionId") REFERENCES "ChatSession"("id")
  ON DELETE CASCADE ON UPDATE CASCADE;
