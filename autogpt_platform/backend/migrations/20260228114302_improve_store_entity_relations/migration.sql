BEGIN;

-- Drop illogical column StoreListing.agentGraphVersion;
-- Update StoreListing:AgentGraph relation to be 1:+ instead of 1:1 (based on agentGraphId)
ALTER TABLE "StoreListing" DROP CONSTRAINT "StoreListing_agentGraphId_agentGraphVersion_fkey";
DROP INDEX "StoreListing_agentGraphId_agentGraphVersion_idx";
ALTER TABLE "StoreListing" DROP COLUMN "agentGraphVersion";
ALTER TABLE "AgentGraph" ADD CONSTRAINT "AgentGraph_id_fkey" FOREIGN KEY ("id") REFERENCES "StoreListing"("agentGraphId") ON DELETE NO ACTION ON UPDATE CASCADE;

-- Add uniqueness constraint to Profile.userId and remove invalid data
--
-- Delete any profiles with null userId (which is invalid and doesn't occur in theory)
DELETE FROM "Profile" WHERE "userId" IS NULL;
--
-- Delete duplicate profiles per userId, keeping the most recently updated one
DELETE FROM "Profile"
WHERE "id" IN (
  SELECT "id" FROM (
    SELECT "id", ROW_NUMBER() OVER (
      PARTITION BY "userId" ORDER BY "updatedAt" DESC, "id" DESC
    ) AS rn
    FROM "Profile"
  ) ranked
  WHERE rn > 1
);
--
-- Add userId uniqueness constraint
ALTER TABLE "Profile" ALTER COLUMN "userId" SET NOT NULL;
CREATE UNIQUE INDEX "Profile_userId_key" ON "Profile"("userId");

-- Add formal relation StoreListing.owningUserId -> Profile.userId
ALTER TABLE "StoreListing" ADD CONSTRAINT "StoreListing_owner_Profile_fkey" FOREIGN KEY ("owningUserId") REFERENCES "Profile"("userId") ON DELETE CASCADE ON UPDATE CASCADE;

COMMIT;
