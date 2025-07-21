-- DropIndex
DROP INDEX "APIKey_key_idx";

-- DropIndex
DROP INDEX "APIKey_prefix_idx";

-- DropIndex
DROP INDEX "APIKey_status_idx";

-- DropIndex
DROP INDEX "idx_agent_graph_execution_agent";

-- DropIndex
DROP INDEX "AnalyticsDetails_type_idx";

-- DropIndex
DROP INDEX "AnalyticsMetrics_userId_idx";

-- DropIndex
DROP INDEX "IntegrationWebhook_userId_idx";

-- DropIndex
DROP INDEX "Profile_username_idx";

-- DropIndex
DROP INDEX "idx_store_listing_review_version";

-- DropIndex
DROP INDEX "User_email_idx";

-- DropIndex
DROP INDEX "User_id_idx";

-- DropIndex
DROP INDEX "UserOnboarding_userId_idx";

-- DropIndex
DROP INDEX IF EXISTS "idx_store_listing_version_status";

-- DropIndex  
DROP INDEX IF EXISTS "idx_slv_agent";

-- DropIndex
DROP INDEX IF EXISTS "idx_store_listing_version_approved_listing";

-- DropIndex
DROP INDEX IF EXISTS "StoreListing_agentId_owningUserId_idx";

-- DropIndex
DROP INDEX IF EXISTS "StoreListing_isDeleted_isApproved_idx";

-- DropIndex
DROP INDEX IF EXISTS "StoreListing_isDeleted_idx";

-- DropIndex
DROP INDEX IF EXISTS "StoreListingVersion_agentId_agentVersion_isDeleted_idx";

-- DropIndex
DROP INDEX IF EXISTS "idx_store_listing_approved";

-- DropIndex
DROP INDEX IF EXISTS "idx_slv_categories_gin";

-- CreateIndex
CREATE INDEX "APIKey_prefix_name_idx" ON "APIKey"("prefix", "name");

-- CreateIndex
CREATE INDEX "AgentGraph_forkedFromId_forkedFromVersion_idx" ON "AgentGraph"("forkedFromId", "forkedFromVersion");

-- CreateIndex
CREATE INDEX "AgentGraphExecution_agentPresetId_idx" ON "AgentGraphExecution"("agentPresetId");

-- CreateIndex
CREATE INDEX "AgentNodeExecution_agentNodeId_executionStatus_idx" ON "AgentNodeExecution"("agentNodeId", "executionStatus");

-- CreateIndex
CREATE INDEX "AgentPreset_agentGraphId_agentGraphVersion_idx" ON "AgentPreset"("agentGraphId", "agentGraphVersion");

-- CreateIndex
CREATE INDEX "AgentPreset_webhookId_idx" ON "AgentPreset"("webhookId");

-- CreateIndex
CREATE INDEX "LibraryAgent_agentGraphId_agentGraphVersion_idx" ON "LibraryAgent"("agentGraphId", "agentGraphVersion");

-- CreateIndex
CREATE INDEX "LibraryAgent_creatorId_idx" ON "LibraryAgent"("creatorId");

-- CreateIndex
CREATE INDEX "StoreListing_agentGraphId_agentGraphVersion_idx" ON "StoreListing"("agentGraphId", "agentGraphVersion");

-- CreateIndex
CREATE INDEX "StoreListingReview_reviewByUserId_idx" ON "StoreListingReview"("reviewByUserId");

-- RenameIndex
ALTER INDEX "analyticsDetails" RENAME TO "AnalyticsDetails_userId_type_idx";
