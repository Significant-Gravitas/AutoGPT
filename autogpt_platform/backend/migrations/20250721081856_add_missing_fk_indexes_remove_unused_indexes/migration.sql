-- DropIndex
DROP INDEX IF EXISTS "APIKey_key_idx";

-- DropIndex
DROP INDEX IF EXISTS "APIKey_prefix_idx";

-- DropIndex
DROP INDEX IF EXISTS "APIKey_status_idx";

-- DropIndex
DROP INDEX IF EXISTS "idx_agent_graph_execution_agent";

-- DropIndex
DROP INDEX IF EXISTS "AnalyticsDetails_type_idx";

-- DropIndex
DROP INDEX IF EXISTS "AnalyticsMetrics_userId_idx";

-- DropIndex
DROP INDEX IF EXISTS "IntegrationWebhook_userId_idx";

-- DropIndex
DROP INDEX IF EXISTS "Profile_username_idx";

-- DropIndex
DROP INDEX IF EXISTS "idx_store_listing_review_version";

-- DropIndex
DROP INDEX IF EXISTS "User_email_idx";

-- DropIndex
DROP INDEX IF EXISTS "User_id_idx";

-- DropIndex
DROP INDEX IF EXISTS "UserOnboarding_userId_idx";

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

-- DropIndex
DROP INDEX IF EXISTS "idx_profile_user";

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

-- CreateIndex (Materialized View Performance Indexes)
CREATE INDEX IF NOT EXISTS "idx_mv_review_stats_rating" ON "mv_review_stats" ("avg_rating" DESC);

-- CreateIndex
CREATE INDEX IF NOT EXISTS "idx_mv_review_stats_count" ON "mv_review_stats" ("review_count" DESC);

-- RenameIndex (only if exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'analyticsDetails') THEN
        ALTER INDEX "analyticsDetails" RENAME TO "AnalyticsDetails_userId_type_idx";
    END IF;
END $$;
