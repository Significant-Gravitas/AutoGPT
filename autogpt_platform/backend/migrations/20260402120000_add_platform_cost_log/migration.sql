-- CreateTable
CREATE TABLE "PlatformCostLog" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT,
    "graphExecId" TEXT,
    "nodeExecId" TEXT,
    "graphId" TEXT,
    "nodeId" TEXT,
    "blockId" TEXT,
    "blockName" TEXT,
    "provider" TEXT NOT NULL,
    "credentialId" TEXT,
    "costMicrodollars" BIGINT,
    "inputTokens" INTEGER,
    "outputTokens" INTEGER,
    "dataSize" INTEGER,
    "duration" DOUBLE PRECISION,
    "model" TEXT,
    "trackingType" TEXT,
    "trackingAmount" DOUBLE PRECISION,
    "metadata" JSONB,

    CONSTRAINT "PlatformCostLog_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "PlatformCostLog_userId_createdAt_idx" ON "PlatformCostLog"("userId", "createdAt");

-- CreateIndex
CREATE INDEX "PlatformCostLog_provider_createdAt_idx" ON "PlatformCostLog"("provider", "createdAt");

-- CreateIndex
CREATE INDEX "PlatformCostLog_createdAt_idx" ON "PlatformCostLog"("createdAt");

-- CreateIndex
CREATE INDEX "PlatformCostLog_graphExecId_idx" ON "PlatformCostLog"("graphExecId");

-- CreateIndex
CREATE INDEX "PlatformCostLog_provider_trackingType_idx" ON "PlatformCostLog"("provider", "trackingType");

-- AddForeignKey
ALTER TABLE "PlatformCostLog" ADD CONSTRAINT "PlatformCostLog_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE SET NULL ON UPDATE CASCADE;
