-- CreateTable
CREATE TABLE "SharedExecutionFile" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "executionId" TEXT NOT NULL,
    "fileId" TEXT NOT NULL,
    "shareToken" TEXT NOT NULL,

    CONSTRAINT "SharedExecutionFile_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "SharedExecutionFile_shareToken_fileId_key" ON "SharedExecutionFile"("shareToken", "fileId");

-- CreateIndex
CREATE INDEX "SharedExecutionFile_shareToken_idx" ON "SharedExecutionFile"("shareToken");

-- CreateIndex
CREATE INDEX "SharedExecutionFile_executionId_idx" ON "SharedExecutionFile"("executionId");

-- AddForeignKey
ALTER TABLE "SharedExecutionFile" ADD CONSTRAINT "SharedExecutionFile_executionId_fkey" FOREIGN KEY ("executionId") REFERENCES "AgentGraphExecution"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "SharedExecutionFile" ADD CONSTRAINT "SharedExecutionFile_fileId_fkey" FOREIGN KEY ("fileId") REFERENCES "UserWorkspaceFile"("id") ON DELETE CASCADE ON UPDATE CASCADE;
