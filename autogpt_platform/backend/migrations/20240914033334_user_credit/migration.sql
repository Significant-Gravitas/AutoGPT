/*
  Warnings:

  - The `executionStatus` column on the `AgentNodeExecution` table would be dropped and recreated. This will lead to data loss if there is data in the column.

*/
-- CreateEnum
CREATE TYPE "AgentExecutionStatus" AS ENUM ('INCOMPLETE', 'QUEUED', 'RUNNING', 'COMPLETED', 'FAILED');

-- CreateEnum
CREATE TYPE "UserBlockCreditType" AS ENUM ('TOP_UP', 'USAGE');

-- AlterTable
ALTER TABLE "AgentGraphExecution" ADD COLUMN     "executionStatus" "AgentExecutionStatus" NOT NULL DEFAULT 'COMPLETED',
ADD COLUMN     "startedAt" TIMESTAMP(3);

-- AlterTable
ALTER TABLE "AgentNodeExecution" DROP COLUMN "executionStatus",
ADD COLUMN     "executionStatus" "AgentExecutionStatus" NOT NULL DEFAULT 'COMPLETED';

-- CreateTable
CREATE TABLE "UserBlockCredit" (
    "transactionKey" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT NOT NULL,
    "blockId" TEXT,
    "amount" INTEGER NOT NULL,
    "type" "UserBlockCreditType" NOT NULL,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "metadata" JSONB,

    CONSTRAINT "UserBlockCredit_pkey" PRIMARY KEY ("transactionKey","userId")
);

-- AddForeignKey
ALTER TABLE "UserBlockCredit" ADD CONSTRAINT "UserBlockCredit_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "UserBlockCredit" ADD CONSTRAINT "UserBlockCredit_blockId_fkey" FOREIGN KEY ("blockId") REFERENCES "AgentBlock"("id") ON DELETE SET NULL ON UPDATE CASCADE;
