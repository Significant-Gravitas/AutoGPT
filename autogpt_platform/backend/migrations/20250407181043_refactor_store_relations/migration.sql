/*
- Rename column StoreListing.agentId      to agentGraphId
- Rename column StoreListing.agentVersion to agentGraphVersion
- Rename column StoreListingVersion.agentId      to agentGraphId
- Rename column StoreListingVersion.agentVersion to agentGraphVersion
*/
ALTER TABLE "StoreListing" RENAME COLUMN "agentId" TO "agentGraphId";
ALTER TABLE "StoreListing" RENAME COLUMN "agentVersion" TO "agentGraphVersion";

ALTER TABLE "StoreListingVersion" RENAME COLUMN "agentId" TO "agentGraphId";
ALTER TABLE "StoreListingVersion" RENAME COLUMN "agentVersion" TO "agentGraphVersion";
