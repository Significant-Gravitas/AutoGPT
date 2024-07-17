/*
  Warnings:

  - The primary key for the `AgentGraph` table will be changed. If it partially fails, the table could be left without primary key constraint.

*/
-- RedefineTables
PRAGMA foreign_keys=OFF;
CREATE TABLE "new_AgentNode" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "agentBlockId" TEXT NOT NULL,
    "agentGraphId" TEXT NOT NULL,
    "agentGraphVersion" INTEGER NOT NULL DEFAULT 1,
    "constantInput" TEXT NOT NULL DEFAULT '{}',
    "metadata" TEXT NOT NULL DEFAULT '{}',
    CONSTRAINT "AgentNode_agentBlockId_fkey" FOREIGN KEY ("agentBlockId") REFERENCES "AgentBlock" ("id") ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT "AgentNode_agentGraphId_agentGraphVersion_fkey" FOREIGN KEY ("agentGraphId", "agentGraphVersion") REFERENCES "AgentGraph" ("id", "version") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_AgentNode" ("agentBlockId", "agentGraphId", "constantInput", "id", "metadata") SELECT "agentBlockId", "agentGraphId", "constantInput", "id", "metadata" FROM "AgentNode";
DROP TABLE "AgentNode";
ALTER TABLE "new_AgentNode" RENAME TO "AgentNode";
CREATE TABLE "new_AgentGraphExecution" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "agentGraphId" TEXT NOT NULL,
    "agentGraphVersion" INTEGER NOT NULL DEFAULT 1,
    CONSTRAINT "AgentGraphExecution_agentGraphId_agentGraphVersion_fkey" FOREIGN KEY ("agentGraphId", "agentGraphVersion") REFERENCES "AgentGraph" ("id", "version") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_AgentGraphExecution" ("agentGraphId", "id") SELECT "agentGraphId", "id" FROM "AgentGraphExecution";
DROP TABLE "AgentGraphExecution";
ALTER TABLE "new_AgentGraphExecution" RENAME TO "AgentGraphExecution";
CREATE TABLE "new_AgentGraph" (
    "id" TEXT NOT NULL,
    "version" INTEGER NOT NULL DEFAULT 1,
    "name" TEXT,
    "description" TEXT,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "isTemplate" BOOLEAN NOT NULL DEFAULT false,

    PRIMARY KEY ("id", "version")
);
INSERT INTO "new_AgentGraph" ("description", "id", "name") SELECT "description", "id", "name" FROM "AgentGraph";
DROP TABLE "AgentGraph";
ALTER TABLE "new_AgentGraph" RENAME TO "AgentGraph";
CREATE TABLE "new_AgentGraphExecutionSchedule" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "agentGraphId" TEXT NOT NULL,
    "agentGraphVersion" INTEGER NOT NULL DEFAULT 1,
    "schedule" TEXT NOT NULL,
    "isEnabled" BOOLEAN NOT NULL DEFAULT true,
    "inputData" TEXT NOT NULL,
    "lastUpdated" DATETIME NOT NULL,
    CONSTRAINT "AgentGraphExecutionSchedule_agentGraphId_agentGraphVersion_fkey" FOREIGN KEY ("agentGraphId", "agentGraphVersion") REFERENCES "AgentGraph" ("id", "version") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_AgentGraphExecutionSchedule" ("agentGraphId", "id", "inputData", "isEnabled", "lastUpdated", "schedule") SELECT "agentGraphId", "id", "inputData", "isEnabled", "lastUpdated", "schedule" FROM "AgentGraphExecutionSchedule";
DROP TABLE "AgentGraphExecutionSchedule";
ALTER TABLE "new_AgentGraphExecutionSchedule" RENAME TO "AgentGraphExecutionSchedule";
CREATE INDEX "AgentGraphExecutionSchedule_isEnabled_idx" ON "AgentGraphExecutionSchedule"("isEnabled");
PRAGMA foreign_key_check;
PRAGMA foreign_keys=ON;
