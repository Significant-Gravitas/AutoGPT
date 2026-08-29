/*
  Warnings:

  - You are about to drop the column `agentGraphParentId` on the `AgentGraph` table. All the data in the column will be lost.

*/
-- DropForeignKey
ALTER TABLE "AgentGraph" DROP CONSTRAINT "AgentGraph_agentGraphParentId_version_fkey";

-- AlterTable
ALTER TABLE "AgentGraph" DROP COLUMN "agentGraphParentId";
