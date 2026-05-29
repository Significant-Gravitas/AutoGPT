-- CreateIndex
CREATE INDEX "AgentGraph_userId_isActive_idx" ON "AgentGraph"("userId", "isActive");

-- CreateIndex
CREATE INDEX "AgentGraphExecution_agentGraphId_agentGraphVersion_idx" ON "AgentGraphExecution"("agentGraphId", "agentGraphVersion");

-- CreateIndex
CREATE INDEX "AgentGraphExecution_userId_idx" ON "AgentGraphExecution"("userId");

-- CreateIndex
CREATE INDEX "AgentNode_agentGraphId_agentGraphVersion_idx" ON "AgentNode"("agentGraphId", "agentGraphVersion");

-- CreateIndex
CREATE INDEX "AgentNode_agentBlockId_idx" ON "AgentNode"("agentBlockId");

-- CreateIndex
CREATE INDEX "AgentNode_webhookId_idx" ON "AgentNode"("webhookId");

-- CreateIndex
CREATE INDEX "AgentNodeExecution_agentGraphExecutionId_idx" ON "AgentNodeExecution"("agentGraphExecutionId");

-- CreateIndex
CREATE INDEX "AgentNodeExecution_agentNodeId_idx" ON "AgentNodeExecution"("agentNodeId");

-- CreateIndex
CREATE INDEX "AgentNodeExecutionInputOutput_referencedByOutputExecId_idx" ON "AgentNodeExecutionInputOutput"("referencedByOutputExecId");

-- CreateIndex
CREATE INDEX "AgentNodeLink_agentNodeSourceId_idx" ON "AgentNodeLink"("agentNodeSourceId");

-- CreateIndex
CREATE INDEX "AgentNodeLink_agentNodeSinkId_idx" ON "AgentNodeLink"("agentNodeSinkId");

-- CreateIndex
CREATE INDEX "AnalyticsMetrics_userId_idx" ON "AnalyticsMetrics"("userId");

-- CreateIndex
CREATE INDEX "IntegrationWebhook_userId_idx" ON "IntegrationWebhook"("userId");

-- CreateIndex
CREATE INDEX "UserBlockCredit_userId_createdAt_idx" ON "UserBlockCredit"("userId", "createdAt");
