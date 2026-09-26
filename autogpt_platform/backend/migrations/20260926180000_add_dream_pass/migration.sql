-- The durable record of each dream pass: one row per pass, written by the
-- sync orchestrator as each step runs and by the batch path at submit and in
-- each result callback. A new table, so nothing existing is rewritten.

-- CreateEnum
CREATE TYPE "DreamPassRoute" AS ENUM ('SYNC', 'ANTHROPIC_BATCH');

-- CreateEnum
CREATE TYPE "DreamPassTrigger" AS ENUM ('CRON', 'ADMIN', 'EVAL');

-- CreateEnum
CREATE TYPE "DreamPassPhase" AS ENUM ('GATHER', 'CONSOLIDATE', 'RECOMBINE', 'SANITIZE', 'APPLY', 'DONE');

-- CreateEnum
CREATE TYPE "DreamPassStatus" AS ENUM ('QUEUED', 'RUNNING', 'SUBMITTED', 'APPLYING', 'COMPLETE', 'ERRORED', 'CANCELLED', 'EXPIRED', 'SKIPPED');

-- CreateTable
CREATE TABLE "DreamPass" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "expertId" TEXT,
    "scopeKey" TEXT NOT NULL,
    "route" "DreamPassRoute" NOT NULL,
    "trigger" "DreamPassTrigger" NOT NULL,
    "phase" "DreamPassPhase" NOT NULL DEFAULT 'GATHER',
    "status" "DreamPassStatus" NOT NULL DEFAULT 'QUEUED',
    "skipReason" TEXT,
    "cancelGeneration" INTEGER NOT NULL DEFAULT 0,
    "providerBatchId" TEXT,
    "leaseToken" TEXT,
    "leaseExpiresAt" TIMESTAMP(3),
    "inputBundle" JSONB,
    "phaseOutputs" JSONB,
    "operations" JSONB,
    "usage" JSONB,
    "windowStart" TIMESTAMP(3),
    "windowEnd" TIMESTAMP(3),
    "error" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "startedAt" TIMESTAMP(3),
    "submittedAt" TIMESTAMP(3),
    "appliedAt" TIMESTAMP(3),
    "completedAt" TIMESTAMP(3),
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "DreamPass_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "DreamPass_scopeKey_status_idx" ON "DreamPass"("scopeKey", "status");

-- CreateIndex
CREATE INDEX "DreamPass_userId_createdAt_idx" ON "DreamPass"("userId", "createdAt");

-- AddForeignKey
ALTER TABLE "DreamPass" ADD CONSTRAINT "DreamPass_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "DreamPass" ADD CONSTRAINT "DreamPass_expertId_fkey" FOREIGN KEY ("expertId") REFERENCES "Expert"("id") ON DELETE CASCADE ON UPDATE CASCADE;
