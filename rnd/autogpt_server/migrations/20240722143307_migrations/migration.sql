-- CreateTable
CREATE TABLE "AgentGraph" (
    "id" TEXT NOT NULL,
    "version" INTEGER NOT NULL DEFAULT 1,
    "name" TEXT,
    "description" TEXT,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "isTemplate" BOOLEAN NOT NULL DEFAULT false,

    CONSTRAINT "AgentGraph_pkey" PRIMARY KEY ("id","version")
);

-- CreateTable
CREATE TABLE "AgentNode" (
    "id" TEXT NOT NULL,
    "agentBlockId" TEXT NOT NULL,
    "agentGraphId" TEXT NOT NULL,
    "agentGraphVersion" INTEGER NOT NULL DEFAULT 1,
    "constantInput" TEXT NOT NULL DEFAULT '{}',
    "metadata" TEXT NOT NULL DEFAULT '{}',

    CONSTRAINT "AgentNode_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "AgentNodeLink" (
    "id" TEXT NOT NULL,
    "agentNodeSourceId" TEXT NOT NULL,
    "sourceName" TEXT NOT NULL,
    "agentNodeSinkId" TEXT NOT NULL,
    "sinkName" TEXT NOT NULL,

    CONSTRAINT "AgentNodeLink_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "AgentBlock" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "inputSchema" TEXT NOT NULL,
    "outputSchema" TEXT NOT NULL,

    CONSTRAINT "AgentBlock_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "AgentGraphExecution" (
    "id" TEXT NOT NULL,
    "agentGraphId" TEXT NOT NULL,
    "agentGraphVersion" INTEGER NOT NULL DEFAULT 1,

    CONSTRAINT "AgentGraphExecution_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "AgentNodeExecution" (
    "id" TEXT NOT NULL,
    "agentGraphExecutionId" TEXT NOT NULL,
    "agentNodeId" TEXT NOT NULL,
    "executionStatus" TEXT NOT NULL,
    "addedTime" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "queuedTime" TIMESTAMP(3),
    "startedTime" TIMESTAMP(3),
    "endedTime" TIMESTAMP(3),

    CONSTRAINT "AgentNodeExecution_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "AgentNodeExecutionInputOutput" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "data" TEXT NOT NULL,
    "time" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "referencedByInputExecId" TEXT,
    "referencedByOutputExecId" TEXT,

    CONSTRAINT "AgentNodeExecutionInputOutput_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "AgentGraphExecutionSchedule" (
    "id" TEXT NOT NULL,
    "agentGraphId" TEXT NOT NULL,
    "agentGraphVersion" INTEGER NOT NULL DEFAULT 1,
    "schedule" TEXT NOT NULL,
    "isEnabled" BOOLEAN NOT NULL DEFAULT true,
    "inputData" TEXT NOT NULL,
    "lastUpdated" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "AgentGraphExecutionSchedule_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "AgentBlock_name_key" ON "AgentBlock"("name");

-- CreateIndex
CREATE INDEX "AgentGraphExecutionSchedule_isEnabled_idx" ON "AgentGraphExecutionSchedule"("isEnabled");

-- AddForeignKey
ALTER TABLE "AgentNode" ADD CONSTRAINT "AgentNode_agentBlockId_fkey" FOREIGN KEY ("agentBlockId") REFERENCES "AgentBlock"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AgentNode" ADD CONSTRAINT "AgentNode_agentGraphId_agentGraphVersion_fkey" FOREIGN KEY ("agentGraphId", "agentGraphVersion") REFERENCES "AgentGraph"("id", "version") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AgentNodeLink" ADD CONSTRAINT "AgentNodeLink_agentNodeSourceId_fkey" FOREIGN KEY ("agentNodeSourceId") REFERENCES "AgentNode"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AgentNodeLink" ADD CONSTRAINT "AgentNodeLink_agentNodeSinkId_fkey" FOREIGN KEY ("agentNodeSinkId") REFERENCES "AgentNode"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AgentGraphExecution" ADD CONSTRAINT "AgentGraphExecution_agentGraphId_agentGraphVersion_fkey" FOREIGN KEY ("agentGraphId", "agentGraphVersion") REFERENCES "AgentGraph"("id", "version") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AgentNodeExecution" ADD CONSTRAINT "AgentNodeExecution_agentGraphExecutionId_fkey" FOREIGN KEY ("agentGraphExecutionId") REFERENCES "AgentGraphExecution"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AgentNodeExecution" ADD CONSTRAINT "AgentNodeExecution_agentNodeId_fkey" FOREIGN KEY ("agentNodeId") REFERENCES "AgentNode"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AgentNodeExecutionInputOutput" ADD CONSTRAINT "AgentNodeExecutionInputOutput_referencedByInputExecId_fkey" FOREIGN KEY ("referencedByInputExecId") REFERENCES "AgentNodeExecution"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AgentNodeExecutionInputOutput" ADD CONSTRAINT "AgentNodeExecutionInputOutput_referencedByOutputExecId_fkey" FOREIGN KEY ("referencedByOutputExecId") REFERENCES "AgentNodeExecution"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AgentGraphExecutionSchedule" ADD CONSTRAINT "AgentGraphExecutionSchedule_agentGraphId_agentGraphVersion_fkey" FOREIGN KEY ("agentGraphId", "agentGraphVersion") REFERENCES "AgentGraph"("id", "version") ON DELETE RESTRICT ON UPDATE CASCADE;
