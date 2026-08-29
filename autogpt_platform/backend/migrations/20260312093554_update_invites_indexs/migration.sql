-- DropIndex
DROP INDEX "InvitedUser_status_idx";
-- DropIndex
DROP INDEX "InvitedUser_tallyStatus_idx";
-- CreateIndex
CREATE INDEX "InvitedUser_createdAt_idx"
ON "InvitedUser"("createdAt");
