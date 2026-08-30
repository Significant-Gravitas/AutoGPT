-- CreateEnum
CREATE TYPE "TaskCreatedByType" AS ENUM ('USER', 'EXPERT', 'SCHEDULE', 'DREAM');

-- CreateEnum
CREATE TYPE "DelegatedTaskStatus" AS ENUM ('QUEUED', 'WORKING', 'WAITING_USER', 'DONE', 'FAILED', 'CANCELLED');

-- CreateEnum
CREATE TYPE "DelegatedTaskAcceptance" AS ENUM ('PENDING', 'ACCEPTED', 'REJECTED');

-- CreateTable
CREATE TABLE "DelegatedTask" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT NOT NULL,
    "ownerId" TEXT,
    "parentTaskId" TEXT,
    "rootTaskId" TEXT,
    "originSessionId" TEXT,
    "createdByType" "TaskCreatedByType" NOT NULL,
    "createdById" TEXT,
    "title" TEXT NOT NULL,
    "spec" TEXT NOT NULL,
    "status" "DelegatedTaskStatus" NOT NULL DEFAULT 'QUEUED',
    "outcomeSummary" TEXT,
    "acceptance" "DelegatedTaskAcceptance" NOT NULL DEFAULT 'PENDING',
    "ancestorExpertIds" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "handoffCount" INTEGER NOT NULL DEFAULT 0,
    "revisionCount" INTEGER NOT NULL DEFAULT 0,
    "spendTotal" INTEGER NOT NULL DEFAULT 0,
    "amendments" JSONB NOT NULL DEFAULT '[]',

    CONSTRAINT "DelegatedTask_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "DelegatedTask_ownerId_status_idx" ON "DelegatedTask"("ownerId", "status");

-- CreateIndex
CREATE INDEX "DelegatedTask_rootTaskId_idx" ON "DelegatedTask"("rootTaskId");

-- CreateIndex
CREATE INDEX "DelegatedTask_originSessionId_idx" ON "DelegatedTask"("originSessionId");

-- CreateIndex
CREATE INDEX "DelegatedTask_userId_status_idx" ON "DelegatedTask"("userId", "status");

-- AlterTable
ALTER TABLE "AgentGraphExecution" ADD COLUMN "delegatedTaskId" TEXT;

-- CreateIndex
CREATE INDEX "AgentGraphExecution_delegatedTaskId_idx" ON "AgentGraphExecution"("delegatedTaskId");

-- AddForeignKey
ALTER TABLE "DelegatedTask" ADD CONSTRAINT "DelegatedTask_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "DelegatedTask" ADD CONSTRAINT "DelegatedTask_ownerId_fkey" FOREIGN KEY ("ownerId") REFERENCES "Expert"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "DelegatedTask" ADD CONSTRAINT "DelegatedTask_parentTaskId_fkey" FOREIGN KEY ("parentTaskId") REFERENCES "DelegatedTask"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "DelegatedTask" ADD CONSTRAINT "DelegatedTask_originSessionId_fkey" FOREIGN KEY ("originSessionId") REFERENCES "ChatSession"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AgentGraphExecution" ADD CONSTRAINT "AgentGraphExecution_delegatedTaskId_fkey" FOREIGN KEY ("delegatedTaskId") REFERENCES "DelegatedTask"("id") ON DELETE SET NULL ON UPDATE CASCADE;
