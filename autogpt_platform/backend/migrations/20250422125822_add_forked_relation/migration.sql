-- AlterTable
ALTER TABLE "AgentGraph"
    ADD COLUMN     "forkedFromId" TEXT,
    ADD COLUMN     "forkedFromVersion" INTEGER;

-- AddForeignKey
ALTER TABLE "AgentGraph" ADD CONSTRAINT "AgentGraph_forkedFromId_forkedFromVersion_fkey" FOREIGN KEY ("forkedFromId", "forkedFromVersion") REFERENCES "AgentGraph"("id", "version") ON DELETE SET NULL ON UPDATE CASCADE;
