/*
  Warnings:

  - You are about to drop the `AgentGraphExecutionSchedule` table. If the table is not empty, all the data it contains will be lost.

*/
-- DropForeignKey
ALTER TABLE "AgentGraphExecutionSchedule" DROP CONSTRAINT "AgentGraphExecutionSchedule_agentGraphId_agentGraphVersion_fkey";

-- DropForeignKey
ALTER TABLE "AgentGraphExecutionSchedule" DROP CONSTRAINT "AgentGraphExecutionSchedule_userId_fkey";

-- DropTable
DROP TABLE "AgentGraphExecutionSchedule";
