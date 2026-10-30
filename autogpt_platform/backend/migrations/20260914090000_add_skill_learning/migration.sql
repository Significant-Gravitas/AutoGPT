-- AlterTable
ALTER TABLE "Expert" ADD COLUMN "learningPausedAt" TIMESTAMP(3);

-- CreateTable
CREATE TABLE "SkillLearningSource" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT NOT NULL,
    "expertId" TEXT,
    "ownerKey" TEXT NOT NULL,
    "sourceKind" TEXT NOT NULL,
    "sourceId" TEXT NOT NULL,
    "revision" TEXT NOT NULL,
    "processedRevision" TEXT,
    "origin" TEXT NOT NULL DEFAULT 'ordinary',
    "eligibility" TEXT NOT NULL DEFAULT 'eligible',
    "epoch" INTEGER NOT NULL DEFAULT 0,
    "approvalEventId" TEXT,
    "approvalActorId" TEXT,
    "approvedRevision" TEXT,
    "evidenceRefs" JSONB NOT NULL DEFAULT '[]',
    "outcomeSignals" JSONB NOT NULL DEFAULT '[]',
    "excludedAt" TIMESTAMP(3),
    "excludedByUserId" TEXT,

    CONSTRAINT "SkillLearningSource_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "SkillLearningReview" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "completedAt" TIMESTAMP(3),
    "userId" TEXT NOT NULL,
    "expertId" TEXT,
    "ownerKey" TEXT NOT NULL,
    "sourceId" TEXT NOT NULL,
    "sourceRevision" TEXT NOT NULL,
    "policyVersion" INTEGER NOT NULL,
    "runId" TEXT NOT NULL,
    "disposition" TEXT NOT NULL,
    "reason" TEXT NOT NULL DEFAULT '',
    "attempts" INTEGER NOT NULL DEFAULT 1,
    "changeFingerprint" TEXT,
    "skillName" TEXT,
    "appliedVersionId" TEXT,
    "model" TEXT,
    "inputTokens" INTEGER NOT NULL DEFAULT 0,
    "outputTokens" INTEGER NOT NULL DEFAULT 0,
    "costMicrodollars" BIGINT,

    CONSTRAINT "SkillLearningReview_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "SkillHead" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT NOT NULL,
    "expertId" TEXT,
    "ownerKey" TEXT NOT NULL,
    "skillName" TEXT NOT NULL,
    "currentVersion" INTEGER NOT NULL DEFAULT 0,
    "currentVersionId" TEXT,
    "contentHash" TEXT,
    "autoImprove" BOOLEAN NOT NULL DEFAULT true,
    "learningPausedAt" TIMESTAMP(3),
    "usePausedAt" TIMESTAMP(3),

    CONSTRAINT "SkillHead_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "SkillVersion" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT NOT NULL,
    "expertId" TEXT,
    "ownerKey" TEXT NOT NULL,
    "skillName" TEXT NOT NULL,
    "headId" TEXT NOT NULL,
    "version" INTEGER NOT NULL,
    "content" TEXT NOT NULL,
    "contentHash" TEXT NOT NULL,
    "description" TEXT NOT NULL,
    "triggers" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "origin" TEXT NOT NULL,
    "actorUserId" TEXT,
    "summary" TEXT NOT NULL DEFAULT '',
    "baseVersionId" TEXT,
    "restoredFromVersionId" TEXT,
    "reviewId" TEXT,
    "sources" JSONB NOT NULL DEFAULT '[]',
    "evidence" JSONB NOT NULL DEFAULT '[]',
    "limits" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "state" TEXT NOT NULL DEFAULT 'ready',
    "stateReason" TEXT NOT NULL DEFAULT '',
    "blockedPatternClass" TEXT,
    "blockedStep" TEXT,

    CONSTRAINT "SkillVersion_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "SkillSuppression" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT NOT NULL,
    "expertId" TEXT,
    "ownerKey" TEXT NOT NULL,
    "skillName" TEXT NOT NULL,
    "behaviorFingerprint" TEXT NOT NULL,
    "behaviorTokens" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "evidenceFingerprints" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "actorUserId" TEXT,
    "reason" TEXT NOT NULL DEFAULT '',

    CONSTRAINT "SkillSuppression_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "SkillUseEvent" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT NOT NULL,
    "expertId" TEXT,
    "ownerKey" TEXT NOT NULL,
    "skillName" TEXT NOT NULL,
    "versionId" TEXT,
    "sessionId" TEXT,
    "kind" TEXT NOT NULL,
    "detail" TEXT NOT NULL DEFAULT '',
    "actorUserId" TEXT,

    CONSTRAINT "SkillUseEvent_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "SkillLearningSource_userId_sourceKind_sourceId_key" ON "SkillLearningSource"("userId", "sourceKind", "sourceId");

-- CreateIndex
CREATE INDEX "SkillLearningSource_userId_ownerKey_eligibility_updatedAt_idx" ON "SkillLearningSource"("userId", "ownerKey", "eligibility", "updatedAt");

-- CreateIndex
CREATE UNIQUE INDEX "SkillLearningReview_sourceId_sourceRevision_policyVersion_key" ON "SkillLearningReview"("sourceId", "sourceRevision", "policyVersion");

-- CreateIndex
CREATE INDEX "SkillLearningReview_userId_ownerKey_createdAt_idx" ON "SkillLearningReview"("userId", "ownerKey", "createdAt");

-- CreateIndex
CREATE INDEX "SkillLearningReview_runId_idx" ON "SkillLearningReview"("runId");

-- CreateIndex
CREATE UNIQUE INDEX "SkillHead_userId_ownerKey_skillName_key" ON "SkillHead"("userId", "ownerKey", "skillName");

-- CreateIndex
CREATE INDEX "SkillHead_userId_ownerKey_idx" ON "SkillHead"("userId", "ownerKey");

-- CreateIndex
CREATE UNIQUE INDEX "SkillVersion_headId_version_key" ON "SkillVersion"("headId", "version");

-- CreateIndex
CREATE INDEX "SkillVersion_userId_ownerKey_skillName_createdAt_idx" ON "SkillVersion"("userId", "ownerKey", "skillName", "createdAt");

-- CreateIndex
CREATE INDEX "SkillVersion_userId_state_createdAt_idx" ON "SkillVersion"("userId", "state", "createdAt");

-- CreateIndex
CREATE UNIQUE INDEX "SkillSuppression_userId_ownerKey_skillName_behaviorFingerprint_key" ON "SkillSuppression"("userId", "ownerKey", "skillName", "behaviorFingerprint");

-- CreateIndex
CREATE INDEX "SkillUseEvent_userId_ownerKey_skillName_createdAt_idx" ON "SkillUseEvent"("userId", "ownerKey", "skillName", "createdAt");

-- CreateIndex
CREATE INDEX "SkillUseEvent_versionId_kind_idx" ON "SkillUseEvent"("versionId", "kind");

-- AddForeignKey
ALTER TABLE "SkillLearningSource" ADD CONSTRAINT "SkillLearningSource_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "SkillLearningReview" ADD CONSTRAINT "SkillLearningReview_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "SkillLearningReview" ADD CONSTRAINT "SkillLearningReview_sourceId_fkey" FOREIGN KEY ("sourceId") REFERENCES "SkillLearningSource"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "SkillHead" ADD CONSTRAINT "SkillHead_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "SkillVersion" ADD CONSTRAINT "SkillVersion_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "SkillVersion" ADD CONSTRAINT "SkillVersion_headId_fkey" FOREIGN KEY ("headId") REFERENCES "SkillHead"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "SkillSuppression" ADD CONSTRAINT "SkillSuppression_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "SkillUseEvent" ADD CONSTRAINT "SkillUseEvent_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
