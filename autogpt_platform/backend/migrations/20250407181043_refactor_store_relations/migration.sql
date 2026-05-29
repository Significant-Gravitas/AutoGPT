/*
- Rename column StoreListing.agentId      to agentGraphId
- Rename column StoreListing.agentVersion to agentGraphVersion
- Rename column StoreListingVersion.agentId      to agentGraphId
- Rename column StoreListingVersion.agentVersion to agentGraphVersion
*/

-- Drop foreign key constraints on columns we're about to rename
ALTER TABLE "StoreListing" DROP CONSTRAINT "StoreListing_agentId_agentVersion_fkey";
ALTER TABLE "StoreListingVersion" DROP CONSTRAINT "StoreListingVersion_agentId_agentVersion_fkey";

-- Drop indices on columns we're about to rename
DROP INDEX "StoreListing_agentId_key";
DROP INDEX "StoreListingVersion_agentId_agentVersion_idx";

-- Rename columns
ALTER TABLE "StoreListing" RENAME COLUMN "agentId" TO "agentGraphId";
ALTER TABLE "StoreListing" RENAME COLUMN "agentVersion" TO "agentGraphVersion";
ALTER TABLE "StoreListingVersion" RENAME COLUMN "agentId" TO "agentGraphId";
ALTER TABLE "StoreListingVersion" RENAME COLUMN "agentVersion" TO "agentGraphVersion";

-- Re-create indices with updated name on renamed columns
CREATE UNIQUE INDEX "StoreListing_agentGraphId_key" ON "StoreListing"("agentGraphId");
CREATE INDEX "StoreListingVersion_agentGraphId_agentGraphVersion_idx" ON "StoreListingVersion"("agentGraphId", "agentGraphVersion");

-- Re-create foreign key constraints with updated name on renamed columns
ALTER TABLE "StoreListing" ADD CONSTRAINT "StoreListing_agentGraphId_agentGraphVersion_fkey"
FOREIGN KEY ("agentGraphId", "agentGraphVersion") REFERENCES "AgentGraph"("id", "version")
ON DELETE CASCADE
ON UPDATE CASCADE;

ALTER TABLE "StoreListingVersion" ADD CONSTRAINT "StoreListingVersion_agentGraphId_agentGraphVersion_fkey"
FOREIGN KEY ("agentGraphId", "agentGraphVersion") REFERENCES "AgentGraph"("id", "version")
ON DELETE RESTRICT
ON UPDATE CASCADE;
