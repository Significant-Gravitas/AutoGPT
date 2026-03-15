-- AlterTable
ALTER TABLE "AgentGraph" ADD COLUMN     "agentGraphParentId" TEXT;

-- AddForeignKey
ALTER TABLE "AgentGraph" ADD CONSTRAINT "AgentGraph_agentGraphParentId_version_fkey" FOREIGN KEY ("agentGraphParentId", "version") REFERENCES "AgentGraph"("id", "version") ON DELETE RESTRICT ON UPDATE CASCADE;
