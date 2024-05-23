-- CreateTable
CREATE TABLE "AgentNode" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "agentComponentId" TEXT NOT NULL,
    "parentNodeId" TEXT,
    CONSTRAINT "AgentNode_parentNodeId_fkey" FOREIGN KEY ("parentNodeId") REFERENCES "AgentNode" ("id") ON DELETE SET NULL ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "AgentNodeExecution" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "agentNodeId" TEXT NOT NULL,
    "inputSchema" TEXT,
    "outputSchema" TEXT,
    "input" TEXT NOT NULL,
    "output" TEXT NOT NULL,
    "executionStatus" TEXT NOT NULL,
    CONSTRAINT "AgentNodeExecution_agentNodeId_fkey" FOREIGN KEY ("agentNodeId") REFERENCES "AgentNode" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "FileDefinition" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "path" TEXT NOT NULL,
    "metadata" TEXT NOT NULL,
    "mimeType" TEXT NOT NULL,
    "size" INTEGER NOT NULL,
    "hash" TEXT NOT NULL,
    "encoding" TEXT NOT NULL
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
CREATE UNIQUE INDEX "_InputFiles_AB_unique" ON "_InputFiles"("A", "B");

-- CreateIndex
CREATE INDEX "_InputFiles_B_index" ON "_InputFiles"("B");

-- CreateIndex
CREATE UNIQUE INDEX "_OutputFiles_AB_unique" ON "_OutputFiles"("A", "B");

-- CreateIndex
CREATE INDEX "_OutputFiles_B_index" ON "_OutputFiles"("B");
