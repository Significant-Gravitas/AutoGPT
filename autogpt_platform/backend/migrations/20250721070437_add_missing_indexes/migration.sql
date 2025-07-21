-- DropIndex
DROP INDEX "idx_agent_graph_execution_agent";

-- DropIndex
DROP INDEX "idx_store_listing_review_version";

-- CreateIndex
CREATE INDEX "AgentGraph_id_idx" ON "AgentGraph"("id");

-- CreateIndex
CREATE INDEX "AgentNodeExecution_agentGraphExecutionId_idx" ON "AgentNodeExecution"("agentGraphExecutionId");

-- CreateIndex
CREATE INDEX "AgentNodeExecutionInputOutput_agentPresetId_idx" ON "AgentNodeExecutionInputOutput"("agentPresetId");

-- CreateIndex
CREATE INDEX "AgentNodeExecutionInputOutput_referencedByInputExecId_idx" ON "AgentNodeExecutionInputOutput"("referencedByInputExecId");

-- CreateIndex
CREATE INDEX "StoreListing_owningUserId_idx" ON "StoreListing"("owningUserId");

-- CreateIndex
CREATE INDEX "StoreListingVersion_storeListingId_idx" ON "StoreListingVersion"("storeListingId");
