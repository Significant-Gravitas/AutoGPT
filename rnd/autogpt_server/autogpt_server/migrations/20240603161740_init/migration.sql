-- CreateTable
CREATE TABLE "AgentGraph" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "name" TEXT,
    "description" TEXT,
    "startingAgentNodeId" TEXT NOT NULL,
    CONSTRAINT "AgentGraph_startingAgentNodeId_fkey" FOREIGN KEY ("startingAgentNodeId") REFERENCES "AgentNode" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "AgentNode" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "agentComponentId" TEXT NOT NULL,
    "agentGraphId" TEXT NOT NULL,
    "parentNodeId" TEXT,
    CONSTRAINT "AgentNode_agentComponentId_fkey" FOREIGN KEY ("agentComponentId") REFERENCES "AgentComponent" ("id") ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT "AgentNode_agentGraphId_fkey" FOREIGN KEY ("agentGraphId") REFERENCES "AgentGraph" ("id") ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT "AgentNode_parentNodeId_fkey" FOREIGN KEY ("parentNodeId") REFERENCES "AgentNode" ("id") ON DELETE SET NULL ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "AgentComponent" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "name" TEXT NOT NULL,
    "code" TEXT NOT NULL,
    "inputSchema" TEXT NOT NULL
);

-- CreateTable
CREATE TABLE "AgentComponentOutput" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "outputName" TEXT NOT NULL,
    "outputSchema" TEXT NOT NULL,
    "description" TEXT NOT NULL,
    "agentComponentId" TEXT NOT NULL,
    CONSTRAINT "AgentComponentOutput_agentComponentId_fkey" FOREIGN KEY ("agentComponentId") REFERENCES "AgentComponent" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "AgentNodeExecution" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "agentNodeId" TEXT NOT NULL,
    "inputData" TEXT NOT NULL,
    "outputData" TEXT NOT NULL,
    "outputTypeId" TEXT,
    "executionStatus" TEXT NOT NULL,
    "executionStateData" TEXT NOT NULL,
    CONSTRAINT "AgentNodeExecution_agentNodeId_fkey" FOREIGN KEY ("agentNodeId") REFERENCES "AgentNode" ("id") ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT "AgentNodeExecution_outputTypeId_fkey" FOREIGN KEY ("outputTypeId") REFERENCES "AgentComponentOutput" ("id") ON DELETE SET NULL ON UPDATE CASCADE
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
CREATE TABLE "_AgentComponentOutputToAgentNode" (
    "A" TEXT NOT NULL,
    "B" TEXT NOT NULL,
    CONSTRAINT "_AgentComponentOutputToAgentNode_A_fkey" FOREIGN KEY ("A") REFERENCES "AgentComponentOutput" ("id") ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "_AgentComponentOutputToAgentNode_B_fkey" FOREIGN KEY ("B") REFERENCES "AgentNode" ("id") ON DELETE CASCADE ON UPDATE CASCADE
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
CREATE UNIQUE INDEX "_AgentComponentOutputToAgentNode_AB_unique" ON "_AgentComponentOutputToAgentNode"("A", "B");

-- CreateIndex
CREATE INDEX "_AgentComponentOutputToAgentNode_B_index" ON "_AgentComponentOutputToAgentNode"("B");

-- CreateIndex
CREATE UNIQUE INDEX "_InputFiles_AB_unique" ON "_InputFiles"("A", "B");

-- CreateIndex
CREATE INDEX "_InputFiles_B_index" ON "_InputFiles"("B");

-- CreateIndex
CREATE UNIQUE INDEX "_OutputFiles_AB_unique" ON "_OutputFiles"("A", "B");

-- CreateIndex
CREATE INDEX "_OutputFiles_B_index" ON "_OutputFiles"("B");
