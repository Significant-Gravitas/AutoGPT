-- A task is one piece of work with its own record, apart from the chat it was
-- asked in. TaskSession links tasks and chats many to many: a chat can carry
-- several tasks and a task can span several chats. Two new tables, so nothing
-- existing is rewritten.

-- CreateEnum
CREATE TYPE "TaskStatus" AS ENUM ('TODO', 'IN_PROGRESS', 'DONE', 'FAILED', 'CANCELLED');

-- CreateEnum
CREATE TYPE "TaskCreator" AS ENUM ('USER', 'OTTO', 'EXPERT');

-- CreateTable
CREATE TABLE "Task" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT NOT NULL,
    "title" TEXT NOT NULL,
    "description" TEXT,
    "status" "TaskStatus" NOT NULL DEFAULT 'TODO',
    "ownerExpertId" TEXT,
    "createdBy" "TaskCreator" NOT NULL,
    "createdByExpertId" TEXT,
    "startedAt" TIMESTAMP(3),
    "completedAt" TIMESTAMP(3),
    "isArchived" BOOLEAN NOT NULL DEFAULT false,
    "metadata" JSONB NOT NULL DEFAULT '{}',
    "organizationId" TEXT,
    "teamId" TEXT,
    "visibility" "ResourceVisibility" NOT NULL DEFAULT 'PRIVATE',

    CONSTRAINT "Task_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "TaskSession" (
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "taskId" TEXT NOT NULL,
    "chatSessionId" TEXT NOT NULL,

    CONSTRAINT "TaskSession_pkey" PRIMARY KEY ("taskId","chatSessionId")
);

-- CreateIndex
CREATE INDEX "Task_userId_isArchived_updatedAt_idx" ON "Task"("userId", "isArchived", "updatedAt");

-- CreateIndex
CREATE INDEX "Task_userId_status_idx" ON "Task"("userId", "status");

-- CreateIndex
CREATE INDEX "Task_ownerExpertId_idx" ON "Task"("ownerExpertId");

-- CreateIndex
CREATE INDEX "Task_teamId_updatedAt_idx" ON "Task"("teamId", "updatedAt");

-- CreateIndex
CREATE INDEX "Task_organizationId_idx" ON "Task"("organizationId");

-- CreateIndex
CREATE INDEX "TaskSession_chatSessionId_idx" ON "TaskSession"("chatSessionId");

-- AddForeignKey
ALTER TABLE "Task" ADD CONSTRAINT "Task_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Task" ADD CONSTRAINT "Task_ownerExpertId_fkey" FOREIGN KEY ("ownerExpertId") REFERENCES "Expert"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Task" ADD CONSTRAINT "Task_createdByExpertId_fkey" FOREIGN KEY ("createdByExpertId") REFERENCES "Expert"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Task" ADD CONSTRAINT "Task_teamId_fkey" FOREIGN KEY ("teamId") REFERENCES "Team"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "TaskSession" ADD CONSTRAINT "TaskSession_taskId_fkey" FOREIGN KEY ("taskId") REFERENCES "Task"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "TaskSession" ADD CONSTRAINT "TaskSession_chatSessionId_fkey" FOREIGN KEY ("chatSessionId") REFERENCES "ChatSession"("id") ON DELETE CASCADE ON UPDATE CASCADE;
