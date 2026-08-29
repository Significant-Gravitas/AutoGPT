-- CreateEnum
CREATE TYPE "ReviewStatus" AS ENUM ('WAITING', 'APPROVED', 'REJECTED');

-- AlterEnum
ALTER TYPE "AgentExecutionStatus" ADD VALUE 'REVIEW';

-- CreateTable
CREATE TABLE "PendingHumanReview" (
    "nodeExecId" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "graphExecId" TEXT NOT NULL,
    "graphId" TEXT NOT NULL,
    "graphVersion" INTEGER NOT NULL,
    "payload" JSONB NOT NULL,
    "instructions" TEXT,
    "editable" BOOLEAN NOT NULL DEFAULT true,
    "status" "ReviewStatus" NOT NULL DEFAULT 'WAITING',
    "reviewMessage" TEXT,
    "wasEdited" BOOLEAN,
    "processed" BOOLEAN NOT NULL DEFAULT false,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3),
    "reviewedAt" TIMESTAMP(3),

    CONSTRAINT "PendingHumanReview_pkey" PRIMARY KEY ("nodeExecId")
);

-- CreateIndex
CREATE INDEX "PendingHumanReview_userId_status_idx" ON "PendingHumanReview"("userId", "status");

-- CreateIndex
CREATE INDEX "PendingHumanReview_graphExecId_status_idx" ON "PendingHumanReview"("graphExecId", "status");

-- CreateIndex
CREATE UNIQUE INDEX "PendingHumanReview_nodeExecId_key" ON "PendingHumanReview"("nodeExecId");

-- AddForeignKey
ALTER TABLE "PendingHumanReview" ADD CONSTRAINT "PendingHumanReview_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "PendingHumanReview" ADD CONSTRAINT "PendingHumanReview_nodeExecId_fkey" FOREIGN KEY ("nodeExecId") REFERENCES "AgentNodeExecution"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "PendingHumanReview" ADD CONSTRAINT "PendingHumanReview_graphExecId_fkey" FOREIGN KEY ("graphExecId") REFERENCES "AgentGraphExecution"("id") ON DELETE CASCADE ON UPDATE CASCADE;
