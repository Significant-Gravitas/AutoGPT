-- DropIndex
DROP INDEX "AgentGraph_userId_isActive_idx";

-- DropIndex
DROP INDEX "AgentGraphExecution_userId_idx";

-- CreateIndex
CREATE INDEX "AgentGraph_userId_isActive_id_version_idx" ON "AgentGraph"("userId", "isActive", "id", "version");

-- CreateIndex
CREATE INDEX "AgentGraphExecution_userId_isDeleted_createdAt_idx" ON "AgentGraphExecution"("userId", "isDeleted", "createdAt");
