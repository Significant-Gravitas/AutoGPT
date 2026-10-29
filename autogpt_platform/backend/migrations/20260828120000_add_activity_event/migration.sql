-- CreateEnum
CREATE TYPE "ActivityEventCategory" AS ENUM ('FILE', 'INTEGRATION', 'RUN', 'SCHEDULE');

-- CreateTable
CREATE TABLE "ActivityEvent" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT NOT NULL,
    "organizationId" TEXT,
    "expertId" TEXT,
    "sessionId" TEXT,
    "graphExecId" TEXT,
    "nodeExecId" TEXT,
    "scheduleId" TEXT,
    "category" "ActivityEventCategory" NOT NULL,
    "eventType" TEXT NOT NULL,
    "provider" TEXT,
    "objectId" TEXT,
    "title" TEXT NOT NULL,
    "data" JSONB NOT NULL DEFAULT '{}',

    CONSTRAINT "ActivityEvent_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "ActivityEvent_userId_createdAt_idx" ON "ActivityEvent"("userId", "createdAt");

-- CreateIndex
CREATE INDEX "ActivityEvent_userId_category_createdAt_idx" ON "ActivityEvent"("userId", "category", "createdAt");

-- CreateIndex
CREATE INDEX "ActivityEvent_sessionId_idx" ON "ActivityEvent"("sessionId");

-- CreateIndex
CREATE INDEX "ActivityEvent_expertId_createdAt_idx" ON "ActivityEvent"("expertId", "createdAt");

-- AddForeignKey
ALTER TABLE "ActivityEvent" ADD CONSTRAINT "ActivityEvent_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
