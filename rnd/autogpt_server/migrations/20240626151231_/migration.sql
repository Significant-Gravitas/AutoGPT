-- CreateTable
CREATE TABLE "AgentGraph" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "name" TEXT,
    "description" TEXT
);

-- CreateTable
CREATE TABLE "AgentNode" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "agentBlockId" TEXT NOT NULL,
    "agentGraphId" TEXT NOT NULL,
    "constantInput" TEXT NOT NULL DEFAULT '{}',
    "metadata" TEXT NOT NULL DEFAULT '{}',
    CONSTRAINT "AgentNode_agentBlockId_fkey" FOREIGN KEY ("agentBlockId") REFERENCES "AgentBlock" ("id") ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT "AgentNode_agentGraphId_fkey" FOREIGN KEY ("agentGraphId") REFERENCES "AgentGraph" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "AgentNodeLink" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "agentNodeSourceId" TEXT NOT NULL,
    "sourceName" TEXT NOT NULL,
    "agentNodeSinkId" TEXT NOT NULL,
    "sinkName" TEXT NOT NULL,
    CONSTRAINT "AgentNodeLink_agentNodeSourceId_fkey" FOREIGN KEY ("agentNodeSourceId") REFERENCES "AgentNode" ("id") ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT "AgentNodeLink_agentNodeSinkId_fkey" FOREIGN KEY ("agentNodeSinkId") REFERENCES "AgentNode" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "AgentBlock" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "name" TEXT NOT NULL,
    "inputSchema" TEXT NOT NULL,
    "outputSchema" TEXT NOT NULL
);

-- CreateTable
CREATE TABLE "AgentNodeExecution" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "executionId" TEXT NOT NULL,
    "agentNodeId" TEXT NOT NULL,
    "inputData" TEXT,
    "outputName" TEXT,
    "outputData" TEXT,
    "executionStatus" TEXT NOT NULL,
    "creationTime" DATETIME NOT NULL,
    "startTime" DATETIME,
    "endTime" DATETIME,
    CONSTRAINT "AgentNodeExecution_agentNodeId_fkey" FOREIGN KEY ("agentNodeId") REFERENCES "AgentNode" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "FileDefinition" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "path" TEXT NOT NULL,
    "metadata" TEXT,
    "mimeType" TEXT,
    "size" INTEGER,
    "hash" TEXT,
    "encoding" TEXT
);

-- CreateTable
CREATE TABLE "AgentExecutionSchedule" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "agentGraphId" TEXT NOT NULL,
    "schedule" TEXT NOT NULL,
    "isEnabled" BOOLEAN NOT NULL DEFAULT true,
    "inputData" TEXT NOT NULL,
    "lastUpdated" DATETIME NOT NULL,
    CONSTRAINT "AgentExecutionSchedule_agentGraphId_fkey" FOREIGN KEY ("agentGraphId") REFERENCES "AgentGraph" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "_InputFiles" (
    "A" TEXT NOT NULL,
    "B" TEXT NOT NULL,
    CONSTRAINT "_InputFiles_A_fkey" FOREIGN KEY ("A") REFERENCES "AgentNodeExecution" ("id") ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "_InputFiles_B_fkey" FOREIGN KEY ("B") REFERENCES "FileDefinition" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "_OutputFiles" (
    "A" TEXT NOT NULL,
    "B" TEXT NOT NULL,
    CONSTRAINT "_OutputFiles_A_fkey" FOREIGN KEY ("A") REFERENCES "AgentNodeExecution" ("id") ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "_OutputFiles_B_fkey" FOREIGN KEY ("B") REFERENCES "FileDefinition" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateIndex
CREATE UNIQUE INDEX "AgentBlock_name_key" ON "AgentBlock"("name");

-- CreateIndex
CREATE INDEX "AgentExecutionSchedule_isEnabled_idx" ON "AgentExecutionSchedule"("isEnabled");

-- CreateIndex
CREATE UNIQUE INDEX "_InputFiles_AB_unique" ON "_InputFiles"("A", "B");

-- CreateIndex
CREATE INDEX "_InputFiles_B_index" ON "_InputFiles"("B");

-- CreateIndex
CREATE UNIQUE INDEX "_OutputFiles_AB_unique" ON "_OutputFiles"("A", "B");

-- CreateIndex
CREATE INDEX "_OutputFiles_B_index" ON "_OutputFiles"("B");
