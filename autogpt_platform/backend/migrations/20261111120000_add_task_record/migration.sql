-- The task record: subtasks, where a task is, and who did what, what it
-- asked, what it made and the proof it happened.

-- CreateEnum
CREATE TYPE "TaskActor" AS ENUM ('OTTO', 'EXPERT', 'USER', 'SYSTEM');

-- CreateEnum
CREATE TYPE "TaskActivityKind" AS ENUM ('CREATED', 'STARTED', 'PROGRESS', 'TOOL', 'DELEGATED', 'ASKED', 'ANSWERED', 'ARTIFACT', 'EVIDENCE', 'STATUS', 'RETRIED', 'FAILED', 'DONE');

-- CreateEnum
CREATE TYPE "TaskAskKind" AS ENUM ('QUESTION', 'DECISION', 'PICK', 'APPROVAL', 'SETUP');

-- CreateEnum
CREATE TYPE "TaskAskStatus" AS ENUM ('OPEN', 'ANSWERED', 'EXPIRED', 'WITHDRAWN');

-- CreateEnum
CREATE TYPE "TaskArtifactKind" AS ENUM ('FILE', 'DOCUMENT', 'LINK', 'WORKFLOW', 'MESSAGE');

-- CreateEnum
CREATE TYPE "TaskEvidenceKind" AS ENUM ('TOOL_RESULT', 'FILE', 'LINK', 'QUOTE', 'CHECK');

-- AlterTable
ALTER TABLE "Task" ADD COLUMN     "attempt" INTEGER NOT NULL DEFAULT 1,
ADD COLUMN     "doneWhen" TEXT,
ADD COLUMN     "failureReason" TEXT,
ADD COLUMN     "goal" TEXT,
ADD COLUMN     "nextStep" TEXT,
ADD COLUMN     "parentTaskId" TEXT,
ADD COLUMN     "result" TEXT,
ADD COLUMN     "statusText" TEXT,
ADD COLUMN     "workSessionId" TEXT;

-- CreateTable
CREATE TABLE "TaskActivity" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "taskId" TEXT NOT NULL,
    "actorType" "TaskActor" NOT NULL,
    "actorExpertId" TEXT,
    "chatSessionId" TEXT,
    "kind" "TaskActivityKind" NOT NULL,
    "text" TEXT NOT NULL,
    "isMilestone" BOOLEAN NOT NULL DEFAULT false,
    "data" JSONB NOT NULL DEFAULT '{}',

    CONSTRAINT "TaskActivity_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "TaskAsk" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "taskId" TEXT NOT NULL,
    "chatSessionId" TEXT,
    "askedByExpertId" TEXT,
    "kind" "TaskAskKind" NOT NULL DEFAULT 'QUESTION',
    "question" TEXT NOT NULL,
    "options" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "defaultGuess" TEXT,
    "status" "TaskAskStatus" NOT NULL DEFAULT 'OPEN',
    "answer" TEXT,
    "answeredAt" TIMESTAMP(3),
    "heldCallId" TEXT,

    CONSTRAINT "TaskAsk_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "TaskArtifact" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "taskId" TEXT NOT NULL,
    "kind" "TaskArtifactKind" NOT NULL DEFAULT 'FILE',
    "title" TEXT NOT NULL,
    "workspaceFileId" TEXT,
    "url" TEXT,
    "createdByExpertId" TEXT,
    "chatSessionId" TEXT,
    "isMissing" BOOLEAN NOT NULL DEFAULT false,

    CONSTRAINT "TaskArtifact_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "TaskEvidence" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "taskId" TEXT NOT NULL,
    "claim" TEXT NOT NULL,
    "kind" "TaskEvidenceKind" NOT NULL,
    "toolCallId" TEXT,
    "chatSessionId" TEXT,
    "artifactId" TEXT,
    "url" TEXT,
    "excerpt" TEXT,
    "verified" BOOLEAN NOT NULL DEFAULT false,
    "addedBy" "TaskActor" NOT NULL DEFAULT 'SYSTEM',

    CONSTRAINT "TaskEvidence_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "TaskActivity_taskId_createdAt_idx" ON "TaskActivity"("taskId", "createdAt");

-- CreateIndex
CREATE INDEX "TaskAsk_taskId_status_idx" ON "TaskAsk"("taskId", "status");

-- CreateIndex
CREATE INDEX "TaskAsk_chatSessionId_status_idx" ON "TaskAsk"("chatSessionId", "status");

-- CreateIndex
CREATE INDEX "TaskArtifact_taskId_createdAt_idx" ON "TaskArtifact"("taskId", "createdAt");

-- CreateIndex
CREATE INDEX "TaskArtifact_workspaceFileId_idx" ON "TaskArtifact"("workspaceFileId");

-- CreateIndex
CREATE INDEX "TaskEvidence_taskId_createdAt_idx" ON "TaskEvidence"("taskId", "createdAt");

-- CreateIndex
CREATE INDEX "Task_parentTaskId_idx" ON "Task"("parentTaskId");

-- CreateIndex
CREATE INDEX "Task_workSessionId_idx" ON "Task"("workSessionId");

-- AddForeignKey
ALTER TABLE "Task" ADD CONSTRAINT "Task_parentTaskId_fkey" FOREIGN KEY ("parentTaskId") REFERENCES "Task"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "TaskActivity" ADD CONSTRAINT "TaskActivity_taskId_fkey" FOREIGN KEY ("taskId") REFERENCES "Task"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "TaskAsk" ADD CONSTRAINT "TaskAsk_taskId_fkey" FOREIGN KEY ("taskId") REFERENCES "Task"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "TaskArtifact" ADD CONSTRAINT "TaskArtifact_taskId_fkey" FOREIGN KEY ("taskId") REFERENCES "Task"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "TaskEvidence" ADD CONSTRAINT "TaskEvidence_taskId_fkey" FOREIGN KEY ("taskId") REFERENCES "Task"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "TaskEvidence" ADD CONSTRAINT "TaskEvidence_artifactId_fkey" FOREIGN KEY ("artifactId") REFERENCES "TaskArtifact"("id") ON DELETE SET NULL ON UPDATE CASCADE;
