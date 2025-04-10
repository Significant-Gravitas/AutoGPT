/*
  Warnings:
  - The relation LibraryAgent:AgentPreset was REMOVED
  - A unique constraint covering the columns `[userId,agentGraphId,agentGraphVersion]` on the table `LibraryAgent` will be added. If there are existing duplicate values, this will fail.
  - The foreign key constraints on AgentPreset and LibraryAgent are being changed from CASCADE to RESTRICT for AgentGraph deletion, which means you cannot delete AgentGraphs that have associated LibraryAgents or AgentPresets.

  Use the following query to check whether these conditions are satisfied:

  -- Check for duplicate LibraryAgent userId + agentGraphId + agentGraphVersion combinations that would violate the new unique constraint
  SELECT   la."userId",
           la."agentId"      as graph_id,
           la."agentVersion" as graph_version,
           COUNT(*)          as multiplicity
  FROM     "LibraryAgent" la
  GROUP BY la."userId",
           la."agentId",
           la."agentVersion"
  HAVING   COUNT(*) > 1;
*/

-- Drop foreign key constraints on columns we're about to rename
ALTER TABLE "AgentPreset"  DROP CONSTRAINT "AgentPreset_agentId_agentVersion_fkey";
ALTER TABLE "LibraryAgent" DROP CONSTRAINT "LibraryAgent_agentId_agentVersion_fkey";
ALTER TABLE "LibraryAgent" DROP CONSTRAINT "LibraryAgent_agentPresetId_fkey";

-- Rename columns in AgentPreset
ALTER TABLE "AgentPreset" RENAME COLUMN "agentId"      TO "agentGraphId";
ALTER TABLE "AgentPreset" RENAME COLUMN "agentVersion" TO "agentGraphVersion";

-- Rename columns in LibraryAgent
ALTER TABLE "LibraryAgent" RENAME COLUMN "agentId"      TO "agentGraphId";
ALTER TABLE "LibraryAgent" RENAME COLUMN "agentVersion" TO "agentGraphVersion";

-- Drop LibraryAgent.agentPresetId column
ALTER TABLE "LibraryAgent" DROP COLUMN "agentPresetId";

-- Replace userId index with unique index on userId + agentGraphId + agentGraphVersion
DROP INDEX "LibraryAgent_userId_idx";
CREATE UNIQUE INDEX "LibraryAgent_userId_agentGraphId_agentGraphVersion_key" ON "LibraryAgent"("userId", "agentGraphId", "agentGraphVersion");

-- Re-add the foreign key constraints with new column names
ALTER TABLE "LibraryAgent" ADD CONSTRAINT "LibraryAgent_agentGraphId_agentGraphVersion_fkey"
FOREIGN KEY ("agentGraphId", "agentGraphVersion") REFERENCES "AgentGraph"("id", "version")
ON DELETE RESTRICT  -- Disallow deleting AgentGraph when still referenced by existing LibraryAgents
ON UPDATE CASCADE;

ALTER TABLE "AgentPreset" ADD CONSTRAINT "AgentPreset_agentGraphId_agentGraphVersion_fkey"
FOREIGN KEY ("agentGraphId", "agentGraphVersion") REFERENCES "AgentGraph"("id", "version")
ON DELETE RESTRICT  -- Disallow deleting AgentGraph when still referenced by existing AgentPresets
ON UPDATE CASCADE;
