-- DropForeignKey
ALTER TABLE "AgentGraph" DROP CONSTRAINT "AgentGraph_agentGraphParentId_version_fkey";

-- AddForeignKey
ALTER TABLE "AgentGraph" ADD CONSTRAINT "AgentGraph_agentGraphParentId_version_fkey" FOREIGN KEY ("agentGraphParentId", "version") REFERENCES "AgentGraph"("id", "version") ON DELETE CASCADE ON UPDATE CASCADE;
