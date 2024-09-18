-- AlterTable
ALTER TABLE "AgentNodeExecution" ADD COLUMN     "executionData" TEXT;

-- AlterTable
ALTER TABLE "AgentNodeLink" ADD COLUMN     "isStatic" BOOLEAN NOT NULL DEFAULT false;
