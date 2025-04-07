/*
- Rename column StoreListing.agentId      to agentGraphId
- Rename column StoreListing.agentVersion to agentGraphVersion
- Rename column StoreListingVersion.agentId      to agentGraphId
- Rename column StoreListingVersion.agentVersion to agentGraphVersion
*/
-- DropForeignKey
ALTER TABLE "AgentPreset" DROP CONSTRAINT "AgentPreset_agentGraphId_agentGraphVersion_fkey";

-- DropForeignKey
ALTER TABLE "StoreListing" DROP CONSTRAINT "StoreListing_agentId_agentVersion_fkey";

-- DropForeignKey
ALTER TABLE "StoreListingVersion" DROP CONSTRAINT "StoreListingVersion_agentId_agentVersion_fkey";

-- DropIndex
DROP INDEX "StoreListing_agentId_key";

-- DropIndex
DROP INDEX "StoreListingVersion_agentId_agentVersion_idx";

-- AlterTable
ALTER TABLE "StoreListing" RENAME COLUMN "agentId" TO "agentGraphId";
ALTER TABLE "StoreListing" RENAME COLUMN "agentVersion" TO "agentGraphVersion";

-- AlterTable
ALTER TABLE "StoreListingVersion" RENAME COLUMN "agentId" TO "agentGraphId";
ALTER TABLE "StoreListingVersion" RENAME COLUMN "agentVersion" TO "agentGraphVersion";

-- CreateIndex
CREATE UNIQUE INDEX "StoreListing_agentGraphId_key" ON "StoreListing"("agentGraphId");

-- CreateIndex
CREATE INDEX "StoreListingVersion_agentGraphId_agentGraphVersion_idx" ON "StoreListingVersion"("agentGraphId", "agentGraphVersion");

-- AddForeignKey
ALTER TABLE "AgentPreset" ADD CONSTRAINT "AgentPreset_agentGraphId_agentGraphVersion_fkey" FOREIGN KEY ("agentGraphId", "agentGraphVersion") REFERENCES "AgentGraph"("id", "version") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "StoreListing" ADD CONSTRAINT "StoreListing_agentGraphId_agentGraphVersion_fkey" FOREIGN KEY ("agentGraphId", "agentGraphVersion") REFERENCES "AgentGraph"("id", "version") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "StoreListingVersion" ADD CONSTRAINT "StoreListingVersion_agentGraphId_agentGraphVersion_fkey" FOREIGN KEY ("agentGraphId", "agentGraphVersion") REFERENCES "AgentGraph"("id", "version") ON DELETE RESTRICT ON UPDATE CASCADE;
