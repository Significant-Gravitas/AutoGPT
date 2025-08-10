/*
  Warnings:

  - A unique constraint covering the columns `[agentId]` on the table `StoreListing` will be added. If there are existing duplicate values, this will fail.

*/
-- DropIndex
DROP INDEX "StoreListing_agentId_idx";

-- DropIndex
DROP INDEX "StoreListing_isApproved_idx";

-- DropIndex
DROP INDEX "StoreListingVersion_agentId_agentVersion_isApproved_idx";

-- CreateIndex
CREATE INDEX "StoreListing_agentId_owningUserId_idx" ON "StoreListing"("agentId", "owningUserId");

-- CreateIndex
CREATE INDEX "StoreListing_isDeleted_isApproved_idx" ON "StoreListing"("isDeleted", "isApproved");

-- CreateIndex
CREATE INDEX "StoreListing_isDeleted_idx" ON "StoreListing"("isDeleted");

-- CreateIndex
CREATE UNIQUE INDEX "StoreListing_agentId_key" ON "StoreListing"("agentId");

-- CreateIndex
CREATE INDEX "StoreListingVersion_agentId_agentVersion_isDeleted_idx" ON "StoreListingVersion"("agentId", "agentVersion", "isDeleted");
