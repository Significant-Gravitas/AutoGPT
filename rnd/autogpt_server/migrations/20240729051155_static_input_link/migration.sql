-- AlterTable
ALTER TABLE "AgentNodeExecution" ADD COLUMN "executionData" TEXT;

-- RedefineTables
PRAGMA foreign_keys=OFF;
CREATE TABLE "new_AgentNodeLink" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "agentNodeSourceId" TEXT NOT NULL,
    "sourceName" TEXT NOT NULL,
    "agentNodeSinkId" TEXT NOT NULL,
    "sinkName" TEXT NOT NULL,
    "isStatic" BOOLEAN NOT NULL DEFAULT false,
    CONSTRAINT "AgentNodeLink_agentNodeSourceId_fkey" FOREIGN KEY ("agentNodeSourceId") REFERENCES "AgentNode" ("id") ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT "AgentNodeLink_agentNodeSinkId_fkey" FOREIGN KEY ("agentNodeSinkId") REFERENCES "AgentNode" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_AgentNodeLink" ("agentNodeSinkId", "agentNodeSourceId", "id", "sinkName", "sourceName") SELECT "agentNodeSinkId", "agentNodeSourceId", "id", "sinkName", "sourceName" FROM "AgentNodeLink";
DROP TABLE "AgentNodeLink";
ALTER TABLE "new_AgentNodeLink" RENAME TO "AgentNodeLink";
PRAGMA foreign_key_check;
PRAGMA foreign_keys=ON;
