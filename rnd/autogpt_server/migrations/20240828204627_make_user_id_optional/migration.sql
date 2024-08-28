-- RedefineTables
PRAGMA foreign_keys=OFF;
CREATE TABLE "new_AgentGraphExecutionSchedule" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "agentGraphId" TEXT NOT NULL,
    "agentGraphVersion" INTEGER NOT NULL DEFAULT 1,
    "schedule" TEXT NOT NULL,
    "isEnabled" BOOLEAN NOT NULL DEFAULT true,
    "inputData" TEXT NOT NULL,
    "lastUpdated" DATETIME NOT NULL,
    "userId" TEXT,
    CONSTRAINT "AgentGraphExecutionSchedule_agentGraphId_agentGraphVersion_fkey" FOREIGN KEY ("agentGraphId", "agentGraphVersion") REFERENCES "AgentGraph" ("id", "version") ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT "AgentGraphExecutionSchedule_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User" ("id") ON DELETE SET NULL ON UPDATE CASCADE
);
INSERT INTO "new_AgentGraphExecutionSchedule" ("agentGraphId", "agentGraphVersion", "id", "inputData", "isEnabled", "lastUpdated", "schedule", "userId") SELECT "agentGraphId", "agentGraphVersion", "id", "inputData", "isEnabled", "lastUpdated", "schedule", "userId" FROM "AgentGraphExecutionSchedule";
DROP TABLE "AgentGraphExecutionSchedule";
ALTER TABLE "new_AgentGraphExecutionSchedule" RENAME TO "AgentGraphExecutionSchedule";
CREATE INDEX "AgentGraphExecutionSchedule_isEnabled_idx" ON "AgentGraphExecutionSchedule"("isEnabled");
PRAGMA foreign_key_check;
PRAGMA foreign_keys=ON;
