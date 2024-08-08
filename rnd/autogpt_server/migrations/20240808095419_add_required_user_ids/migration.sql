-- RedefineTables
PRAGMA foreign_keys=OFF;

CREATE TABLE "new_AgentGraphExecution" (
    "id"                TEXT NOT NULL PRIMARY KEY,
    "agentGraphId"      TEXT NOT NULL,
    "agentGraphVersion" INTEGER NOT NULL DEFAULT 1,
    "userId"            TEXT NOT NULL,
    CONSTRAINT "AgentGraphExecution_agentGraphId_agentGraphVersion_fkey"
        FOREIGN KEY ("agentGraphId", "agentGraphVersion")
        REFERENCES "AgentGraph" ("id", "version")
        ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT "AgentGraphExecution_userId_fkey"
        FOREIGN KEY ("userId")
        REFERENCES "User" ("id")
        ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_AgentGraphExecution" ("agentGraphId", "agentGraphVersion", "id", "userId")
SELECT      "agentGraphId",
            "agentGraphVersion",
            "id",
            CASE WHEN "userId" IS NULL THEN '3e53486c-cf57-477e-ba2a-cb02dc828e1a' ELSE "userId" END
FROM        "AgentGraphExecution";
DROP TABLE  "AgentGraphExecution";
ALTER TABLE "new_AgentGraphExecution" RENAME TO "AgentGraphExecution";

CREATE TABLE "new_AgentGraph" (
    "id"                 TEXT NOT NULL,
    "version"            INTEGER NOT NULL DEFAULT 1,
    "name"               TEXT,
    "description"        TEXT,
    "isActive"           BOOLEAN NOT NULL DEFAULT true,
    "isTemplate"         BOOLEAN NOT NULL DEFAULT false,
    "userId"             TEXT NOT NULL,
    "agentGraphParentId" TEXT,
    PRIMARY KEY ("id", "version"),
    CONSTRAINT "AgentGraph_userId_fkey"
        FOREIGN KEY ("userId")
        REFERENCES "User" ("id")
        ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT "AgentGraph_agentGraphParentId_version_fkey"
        FOREIGN KEY ("agentGraphParentId", "version")
        REFERENCES "AgentGraph" ("id", "version")
        ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_AgentGraph" ("agentGraphParentId", "description", "id", "isActive", "isTemplate", "name", "userId", "version")
SELECT      "agentGraphParentId",
            "description",
            "id",
            "isActive",
            "isTemplate",
            "name",
            CASE WHEN "userId" IS NULL THEN '3e53486c-cf57-477e-ba2a-cb02dc828e1a' ELSE "userId" END,
            "version"
FROM        "AgentGraph";
DROP TABLE  "AgentGraph";
ALTER TABLE "new_AgentGraph" RENAME TO "AgentGraph";

CREATE TABLE "new_AgentGraphExecutionSchedule" (
    "id"                TEXT NOT NULL PRIMARY KEY,
    "agentGraphId"      TEXT NOT NULL,
    "agentGraphVersion" INTEGER NOT NULL DEFAULT 1,
    "schedule"          TEXT NOT NULL,
    "isEnabled"         BOOLEAN NOT NULL DEFAULT true,
    "inputData"         TEXT NOT NULL,
    "lastUpdated"       DATETIME NOT NULL,
    "userId"            TEXT NOT NULL,
    CONSTRAINT "AgentGraphExecutionSchedule_agentGraphId_agentGraphVersion_fkey"
        FOREIGN KEY ("agentGraphId", "agentGraphVersion")
        REFERENCES "AgentGraph" ("id", "version")
        ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT "AgentGraphExecutionSchedule_userId_fkey"
        FOREIGN KEY ("userId")
        REFERENCES "User" ("id")
        ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO  "new_AgentGraphExecutionSchedule" ("agentGraphId", "agentGraphVersion", "id", "inputData", "isEnabled", "lastUpdated", "schedule", "userId") 
SELECT       "agentGraphId",
             "agentGraphVersion",
             "id",
             "inputData",
             "isEnabled",
             "lastUpdated",
             "schedule",
             CASE WHEN "userId" IS NULL THEN '3e53486c-cf57-477e-ba2a-cb02dc828e1a' ELSE "userId" END
FROM         "AgentGraphExecutionSchedule";
DROP TABLE   "AgentGraphExecutionSchedule";
ALTER TABLE  "new_AgentGraphExecutionSchedule" RENAME TO "AgentGraphExecutionSchedule";
CREATE INDEX "AgentGraphExecutionSchedule_isEnabled_idx" ON "AgentGraphExecutionSchedule"("isEnabled");

PRAGMA foreign_key_check;
PRAGMA foreign_keys=ON;
