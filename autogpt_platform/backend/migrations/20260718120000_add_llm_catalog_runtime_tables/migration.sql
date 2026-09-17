-- AlterTable
ALTER TABLE "ChatMessage" ADD COLUMN     "model" TEXT,
ADD COLUMN     "routingSource" TEXT;

-- CreateTable
CREATE TABLE "LlmModelMigration" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "sourceModelSlug" TEXT NOT NULL,
    "targetModelSlug" TEXT NOT NULL,
    "reason" TEXT,
    "migratedNodeIds" JSONB NOT NULL DEFAULT '[]',
    "nodeCount" INTEGER NOT NULL,
    "isReverted" BOOLEAN NOT NULL DEFAULT false,
    "revertedAt" TIMESTAMP(3),

    CONSTRAINT "LlmModelMigration_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "LlmModelMigration_targetModelSlug_idx" ON "LlmModelMigration"("targetModelSlug");

-- CreateIndex
CREATE INDEX "LlmModelMigration_sourceModelSlug_isReverted_idx" ON "LlmModelMigration"("sourceModelSlug", "isReverted");


-- CreateIndex (partial unique to prevent multiple active migrations per source)
CREATE UNIQUE INDEX "LlmModelMigration_active_source_key" ON "LlmModelMigration"("sourceModelSlug") WHERE "isReverted" = false;

-- AddCheckConstraint
ALTER TABLE "LlmModelMigration"
    ADD CONSTRAINT "LlmModelMigration_nodeCount_check" CHECK ("nodeCount" >= 0);
