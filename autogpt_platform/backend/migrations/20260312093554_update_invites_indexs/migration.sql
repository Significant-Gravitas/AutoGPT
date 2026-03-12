-- DropIndex
DROP INDEX "InvitedUser_status_idx";
-- DropIndex
DROP INDEX "InvitedUser_tallyStatus_idx";
-- DropIndex
DROP INDEX "UnifiedContentEmbedding_search_idx";
-- CreateIndex
CREATE INDEX "InvitedUser_createdAt_idx"
ON "InvitedUser"("createdAt");
