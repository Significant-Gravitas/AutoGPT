/*
  Warnings:

  - A unique constraint covering the columns `[referencedByInputExecId,referencedByOutputExecId,name]` on the table `AgentNodeExecutionInputOutput` will be added. If there are existing duplicate values, this will fail.

*/
-- CreateIndex
CREATE UNIQUE INDEX "AgentNodeExecutionInputOutput_referencedByInputExecId_refer_key" ON "AgentNodeExecutionInputOutput"("referencedByInputExecId", "referencedByOutputExecId", "name");
