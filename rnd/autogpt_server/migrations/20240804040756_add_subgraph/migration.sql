-- RedefineTables
PRAGMA foreign_keys=OFF;
CREATE TABLE "new_AgentGraph" (
    "id" TEXT NOT NULL,
    "version" INTEGER NOT NULL DEFAULT 1,
    "name" TEXT,
    "description" TEXT,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "isTemplate" BOOLEAN NOT NULL DEFAULT false,
    "agentGraphParentId" TEXT,

    PRIMARY KEY ("id", "version"),
    CONSTRAINT "AgentGraph_agentGraphParentId_version_fkey" FOREIGN KEY ("agentGraphParentId", "version") REFERENCES "AgentGraph" ("id", "version") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_AgentGraph" ("description", "id", "isActive", "isTemplate", "name", "version") SELECT "description", "id", "isActive", "isTemplate", "name", "version" FROM "AgentGraph";
DROP TABLE "AgentGraph";
ALTER TABLE "new_AgentGraph" RENAME TO "AgentGraph";
PRAGMA foreign_key_check;
PRAGMA foreign_keys=ON;
