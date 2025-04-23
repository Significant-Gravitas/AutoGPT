-- DropForeignKey
ALTER TABLE "AgentGraph" DROP CONSTRAINT "AgentGraph_userId_fkey";

-- DropForeignKey
ALTER TABLE "AgentGraphExecution" DROP CONSTRAINT "AgentGraphExecution_agentGraphId_agentGraphVersion_fkey";

-- DropForeignKey
ALTER TABLE "AgentGraphExecution" DROP CONSTRAINT "AgentGraphExecution_userId_fkey";

-- DropForeignKey
ALTER TABLE "AgentGraphExecutionSchedule" DROP CONSTRAINT "AgentGraphExecutionSchedule_agentGraphId_agentGraphVersion_fkey";

-- DropForeignKey
ALTER TABLE "AgentGraphExecutionSchedule" DROP CONSTRAINT "AgentGraphExecutionSchedule_userId_fkey";

-- DropForeignKey
ALTER TABLE "AgentNode" DROP CONSTRAINT "AgentNode_agentGraphId_agentGraphVersion_fkey";

-- DropForeignKey
ALTER TABLE "AgentNodeExecution" DROP CONSTRAINT "AgentNodeExecution_agentGraphExecutionId_fkey";

-- DropForeignKey
ALTER TABLE "AgentNodeExecution" DROP CONSTRAINT "AgentNodeExecution_agentNodeId_fkey";

-- DropForeignKey
ALTER TABLE "AgentNodeExecutionInputOutput" DROP CONSTRAINT "AgentNodeExecutionInputOutput_referencedByInputExecId_fkey";

-- DropForeignKey
ALTER TABLE "AgentNodeExecutionInputOutput" DROP CONSTRAINT "AgentNodeExecutionInputOutput_referencedByOutputExecId_fkey";

-- DropForeignKey
ALTER TABLE "AgentNodeLink" DROP CONSTRAINT "AgentNodeLink_agentNodeSinkId_fkey";

-- DropForeignKey
ALTER TABLE "AgentNodeLink" DROP CONSTRAINT "AgentNodeLink_agentNodeSourceId_fkey";

-- DropForeignKey
ALTER TABLE "AnalyticsDetails" DROP CONSTRAINT "AnalyticsDetails_userId_fkey";

-- DropForeignKey
ALTER TABLE "AnalyticsMetrics" DROP CONSTRAINT "AnalyticsMetrics_userId_fkey";

-- DropForeignKey
ALTER TABLE "UserBlockCredit" DROP CONSTRAINT "UserBlockCredit_userId_fkey";

-- AddForeignKey
ALTER TABLE "AgentGraph" ADD CONSTRAINT "AgentGraph_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AgentNode" ADD CONSTRAINT "AgentNode_agentGraphId_agentGraphVersion_fkey" FOREIGN KEY ("agentGraphId", "agentGraphVersion") REFERENCES "AgentGraph"("id", "version") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AgentNodeLink" ADD CONSTRAINT "AgentNodeLink_agentNodeSourceId_fkey" FOREIGN KEY ("agentNodeSourceId") REFERENCES "AgentNode"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AgentNodeLink" ADD CONSTRAINT "AgentNodeLink_agentNodeSinkId_fkey" FOREIGN KEY ("agentNodeSinkId") REFERENCES "AgentNode"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AgentGraphExecution" ADD CONSTRAINT "AgentGraphExecution_agentGraphId_agentGraphVersion_fkey" FOREIGN KEY ("agentGraphId", "agentGraphVersion") REFERENCES "AgentGraph"("id", "version") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AgentGraphExecution" ADD CONSTRAINT "AgentGraphExecution_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AgentNodeExecution" ADD CONSTRAINT "AgentNodeExecution_agentGraphExecutionId_fkey" FOREIGN KEY ("agentGraphExecutionId") REFERENCES "AgentGraphExecution"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AgentNodeExecution" ADD CONSTRAINT "AgentNodeExecution_agentNodeId_fkey" FOREIGN KEY ("agentNodeId") REFERENCES "AgentNode"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AgentNodeExecutionInputOutput" ADD CONSTRAINT "AgentNodeExecutionInputOutput_referencedByInputExecId_fkey" FOREIGN KEY ("referencedByInputExecId") REFERENCES "AgentNodeExecution"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AgentNodeExecutionInputOutput" ADD CONSTRAINT "AgentNodeExecutionInputOutput_referencedByOutputExecId_fkey" FOREIGN KEY ("referencedByOutputExecId") REFERENCES "AgentNodeExecution"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AgentGraphExecutionSchedule" ADD CONSTRAINT "AgentGraphExecutionSchedule_agentGraphId_agentGraphVersion_fkey" FOREIGN KEY ("agentGraphId", "agentGraphVersion") REFERENCES "AgentGraph"("id", "version") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AgentGraphExecutionSchedule" ADD CONSTRAINT "AgentGraphExecutionSchedule_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AnalyticsDetails" ADD CONSTRAINT "AnalyticsDetails_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AnalyticsMetrics" ADD CONSTRAINT "AnalyticsMetrics_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "UserBlockCredit" ADD CONSTRAINT "UserBlockCredit_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
