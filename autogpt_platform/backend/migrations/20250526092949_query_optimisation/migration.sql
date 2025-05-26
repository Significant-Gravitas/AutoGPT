-- CreateIndex
CREATE INDEX "AgentNodeExecution_agentNodeId_agentGraphExecutionId_execut_idx" ON "AgentNodeExecution"("agentNodeId", "agentGraphExecutionId", "executionStatus");

-- CreateIndex
CREATE INDEX "AgentNodeExecution_agentGraphExecutionId_executionStatus_idx" ON "AgentNodeExecution"("agentGraphExecutionId", "executionStatus");

-- CreateIndex
CREATE INDEX "AgentNodeExecution_agentGraphExecutionId_queuedTime_addedTi_idx" ON "AgentNodeExecution"("agentGraphExecutionId", "queuedTime", "addedTime");

-- CreateIndex
CREATE INDEX "AgentNodeExecution_queuedTime_addedTime_idx" ON "AgentNodeExecution"("queuedTime", "addedTime");

-- CreateIndex
CREATE INDEX "AgentNodeExecutionInputOutput_referencedByInputExecId_name_idx" ON "AgentNodeExecutionInputOutput"("referencedByInputExecId", "name");

-- CreateIndex
CREATE INDEX "NotificationEvent_userNotificationBatchId_idx" ON "NotificationEvent"("userNotificationBatchId");
