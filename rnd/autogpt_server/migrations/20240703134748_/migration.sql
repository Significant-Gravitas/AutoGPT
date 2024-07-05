/*
  Warnings:

  - You are about to drop the `AgentExecutionSchedule` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `FileDefinition` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `_InputFiles` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `_OutputFiles` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the column `creationTime` on the `AgentNodeExecution` table. All the data in the column will be lost.
  - You are about to drop the column `endTime` on the `AgentNodeExecution` table. All the data in the column will be lost.
  - You are about to drop the column `executionId` on the `AgentNodeExecution` table. All the data in the column will be lost.
  - You are about to drop the column `inputData` on the `AgentNodeExecution` table. All the data in the column will be lost.
  - You are about to drop the column `outputData` on the `AgentNodeExecution` table. All the data in the column will be lost.
  - You are about to drop the column `outputName` on the `AgentNodeExecution` table. All the data in the column will be lost.
  - You are about to drop the column `startTime` on the `AgentNodeExecution` table. All the data in the column will be lost.
  - Added the required column `agentGraphExecutionId` to the `AgentNodeExecution` table without a default value. This is not possible if the table is not empty.

*/
-- DropIndex
DROP INDEX "AgentExecutionSchedule_isEnabled_idx";

-- DropIndex
DROP INDEX "_InputFiles_B_index";

-- DropIndex
DROP INDEX "_InputFiles_AB_unique";

-- DropIndex
DROP INDEX "_OutputFiles_B_index";

-- DropIndex
DROP INDEX "_OutputFiles_AB_unique";

-- DropTable
PRAGMA foreign_keys=off;
DROP TABLE "AgentExecutionSchedule";
PRAGMA foreign_keys=on;

-- DropTable
PRAGMA foreign_keys=off;
DROP TABLE "FileDefinition";
PRAGMA foreign_keys=on;

-- DropTable
PRAGMA foreign_keys=off;
DROP TABLE "_InputFiles";
PRAGMA foreign_keys=on;

-- DropTable
PRAGMA foreign_keys=off;
DROP TABLE "_OutputFiles";
PRAGMA foreign_keys=on;

-- CreateTable
CREATE TABLE "AgentGraphExecution" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "agentGraphId" TEXT NOT NULL,
    CONSTRAINT "AgentGraphExecution_agentGraphId_fkey" FOREIGN KEY ("agentGraphId") REFERENCES "AgentGraph" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "AgentNodeExecutionInputOutput" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "name" TEXT NOT NULL,
    "data" TEXT NOT NULL,
    "time" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "referencedByInputExecId" TEXT,
    "referencedByOutputExecId" TEXT,
    CONSTRAINT "AgentNodeExecutionInputOutput_referencedByInputExecId_fkey" FOREIGN KEY ("referencedByInputExecId") REFERENCES "AgentNodeExecution" ("id") ON DELETE SET NULL ON UPDATE CASCADE,
    CONSTRAINT "AgentNodeExecutionInputOutput_referencedByOutputExecId_fkey" FOREIGN KEY ("referencedByOutputExecId") REFERENCES "AgentNodeExecution" ("id") ON DELETE SET NULL ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "AgentGraphExecutionSchedule" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "agentGraphId" TEXT NOT NULL,
    "schedule" TEXT NOT NULL,
    "isEnabled" BOOLEAN NOT NULL DEFAULT true,
    "inputData" TEXT NOT NULL,
    "lastUpdated" DATETIME NOT NULL,
    CONSTRAINT "AgentGraphExecutionSchedule_agentGraphId_fkey" FOREIGN KEY ("agentGraphId") REFERENCES "AgentGraph" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- RedefineTables
PRAGMA foreign_keys=OFF;
CREATE TABLE "new_AgentNodeExecution" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "agentGraphExecutionId" TEXT NOT NULL,
    "agentNodeId" TEXT NOT NULL,
    "executionStatus" TEXT NOT NULL,
    "addedTime" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "queuedTime" DATETIME,
    "startedTime" DATETIME,
    "endedTime" DATETIME,
    CONSTRAINT "AgentNodeExecution_agentGraphExecutionId_fkey" FOREIGN KEY ("agentGraphExecutionId") REFERENCES "AgentGraphExecution" ("id") ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT "AgentNodeExecution_agentNodeId_fkey" FOREIGN KEY ("agentNodeId") REFERENCES "AgentNode" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_AgentNodeExecution" ("agentNodeId", "executionStatus", "id") SELECT "agentNodeId", "executionStatus", "id" FROM "AgentNodeExecution";
DROP TABLE "AgentNodeExecution";
ALTER TABLE "new_AgentNodeExecution" RENAME TO "AgentNodeExecution";
PRAGMA foreign_key_check;
PRAGMA foreign_keys=ON;

-- CreateIndex
CREATE INDEX "AgentGraphExecutionSchedule_isEnabled_idx" ON "AgentGraphExecutionSchedule"("isEnabled");
