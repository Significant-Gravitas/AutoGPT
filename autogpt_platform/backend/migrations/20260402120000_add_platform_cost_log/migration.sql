-- CreateTable
CREATE TABLE "PlatformCostLog" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT NOT NULL,
    "graphExecId" TEXT,
    "nodeExecId" TEXT,
    "graphId" TEXT,
    "nodeId" TEXT,
    "blockId" TEXT NOT NULL,
    "blockName" TEXT NOT NULL,
    "provider" TEXT NOT NULL,
    "credentialId" TEXT NOT NULL,
    "costMicrodollars" INTEGER,
    "inputTokens" INTEGER,
    "outputTokens" INTEGER,
    "dataSize" INTEGER,
    "duration" DOUBLE PRECISION,
    "model" TEXT,
    "metadata" JSONB,

    CONSTRAINT "PlatformCostLog_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "PlatformCostLog_userId_createdAt_idx" ON "PlatformCostLog"("userId", "createdAt");

-- CreateIndex
CREATE INDEX "PlatformCostLog_provider_createdAt_idx" ON "PlatformCostLog"("provider", "createdAt");

-- CreateIndex
CREATE INDEX "PlatformCostLog_graphExecId_idx" ON "PlatformCostLog"("graphExecId");

-- AddForeignKey
ALTER TABLE "PlatformCostLog" ADD CONSTRAINT "PlatformCostLog_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
