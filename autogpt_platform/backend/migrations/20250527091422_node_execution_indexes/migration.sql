-- DropIndex
DROP INDEX "AgentNodeExecution_addedTime_idx";

-- DropIndex
DROP INDEX "AgentNodeExecution_agentGraphExecutionId_idx";

-- DropIndex
DROP INDEX "AgentNodeExecution_agentNodeId_idx";

-- CreateIndex
CREATE INDEX "AgentNodeExecution_agentGraphExecutionId_agentNodeId_execut_idx" ON "AgentNodeExecution"("agentGraphExecutionId", "agentNodeId", "executionStatus");

-- CreateIndex
CREATE INDEX "AgentNodeExecution_addedTime_queuedTime_idx" ON "AgentNodeExecution"("addedTime", "queuedTime");

-- CreateIndex
CREATE INDEX "AgentNodeExecutionInputOutput_name_time_idx" ON "AgentNodeExecutionInputOutput"("name", "time");

-- CreateIndex
CREATE INDEX "NotificationEvent_userNotificationBatchId_idx" ON "NotificationEvent"("userNotificationBatchId");
