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
    "customCreditCost" INTEGER,
    "isReverted" BOOLEAN NOT NULL DEFAULT false,
    "revertedAt" TIMESTAMP(3),

    CONSTRAINT "LlmModelMigration_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "LlmModelMigration_sourceModelSlug_idx" ON "LlmModelMigration"("sourceModelSlug");

-- CreateIndex
CREATE INDEX "LlmModelMigration_targetModelSlug_idx" ON "LlmModelMigration"("targetModelSlug");

-- CreateIndex
CREATE INDEX "LlmModelMigration_isReverted_idx" ON "LlmModelMigration"("isReverted");
