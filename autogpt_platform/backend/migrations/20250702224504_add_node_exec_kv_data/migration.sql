-- CreateTable
CREATE TABLE "AgentNodeExecutionKeyValueData" (
    "userId" TEXT NOT NULL,
    "key" TEXT NOT NULL,
    "agentNodeExecutionId" TEXT NOT NULL,
    "data" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3),

    CONSTRAINT "AgentNodeExecutionKeyValueData_pkey" PRIMARY KEY ("userId","key")
);
