-- CreateSchema (idempotent - safe if already exists)
CREATE SCHEMA IF NOT EXISTS "platform";

-- CreateEnum
CREATE TYPE "LlmCostUnit" AS ENUM ('RUN', 'TOKENS');

-- CreateTable
CREATE TABLE "LlmProvider" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "name" TEXT NOT NULL,
    "displayName" TEXT NOT NULL,
    "description" TEXT,
    "defaultCredentialProvider" TEXT,
    "defaultCredentialId" TEXT,
    "defaultCredentialType" TEXT,
    "metadata" JSONB NOT NULL DEFAULT '{}',

    CONSTRAINT "LlmProvider_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LlmModelCreator" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "name" TEXT NOT NULL,
    "displayName" TEXT NOT NULL,
    "description" TEXT,
    "websiteUrl" TEXT,
    "logoUrl" TEXT,
    "metadata" JSONB NOT NULL DEFAULT '{}',

    CONSTRAINT "LlmModelCreator_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LlmModel" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "slug" TEXT NOT NULL,
    "displayName" TEXT NOT NULL,
    "description" TEXT,
    "providerId" TEXT NOT NULL,
    "creatorId" TEXT,
    "contextWindow" INTEGER NOT NULL,
    "maxOutputTokens" INTEGER,
    "priceTier" INTEGER NOT NULL DEFAULT 1,
    "isEnabled" BOOLEAN NOT NULL DEFAULT true,
    "isRecommended" BOOLEAN NOT NULL DEFAULT false,
    "supportsTools" BOOLEAN NOT NULL DEFAULT false,
    "supportsJsonOutput" BOOLEAN NOT NULL DEFAULT false,
    "supportsReasoning" BOOLEAN NOT NULL DEFAULT false,
    "supportsParallelToolCalls" BOOLEAN NOT NULL DEFAULT false,
    "capabilities" JSONB NOT NULL DEFAULT '{}',
    "metadata" JSONB NOT NULL DEFAULT '{}',

    CONSTRAINT "LlmModel_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LlmModelCost" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "unit" "LlmCostUnit" NOT NULL DEFAULT 'RUN',
    "creditCost" INTEGER NOT NULL,
    "credentialProvider" TEXT NOT NULL,
    "credentialId" TEXT,
    "credentialType" TEXT,
    "currency" TEXT,
    "metadata" JSONB NOT NULL DEFAULT '{}',
    "llmModelId" TEXT NOT NULL,

    CONSTRAINT "LlmModelCost_pkey" PRIMARY KEY ("id")
);

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
CREATE UNIQUE INDEX "LlmProvider_name_key" ON "LlmProvider"("name");

-- CreateIndex
CREATE UNIQUE INDEX "LlmModelCreator_name_key" ON "LlmModelCreator"("name");

-- CreateIndex
CREATE UNIQUE INDEX "LlmModel_slug_key" ON "LlmModel"("slug");

-- CreateIndex
CREATE INDEX "LlmModel_providerId_isEnabled_idx" ON "LlmModel"("providerId", "isEnabled");

-- CreateIndex
CREATE INDEX "LlmModel_creatorId_idx" ON "LlmModel"("creatorId");

-- CreateIndex
CREATE INDEX "LlmModelCost_credentialProvider_idx" ON "LlmModelCost"("credentialProvider");

-- CreateIndex (partial unique for default costs - no specific credential)
CREATE UNIQUE INDEX "LlmModelCost_default_cost_key" ON "LlmModelCost"("llmModelId", "credentialProvider", "unit") WHERE "credentialId" IS NULL;

-- CreateIndex (partial unique for credential-specific costs)
CREATE UNIQUE INDEX "LlmModelCost_credential_cost_key" ON "LlmModelCost"("llmModelId", "credentialProvider", "credentialId", "unit") WHERE "credentialId" IS NOT NULL;

-- CreateIndex
CREATE INDEX "LlmModelMigration_targetModelSlug_idx" ON "LlmModelMigration"("targetModelSlug");

-- CreateIndex
CREATE INDEX "LlmModelMigration_isReverted_idx" ON "LlmModelMigration"("isReverted");

-- CreateIndex
CREATE INDEX "LlmModelMigration_sourceModelSlug_isReverted_idx" ON "LlmModelMigration"("sourceModelSlug", "isReverted");

-- CreateIndex (partial unique to prevent multiple active migrations per source)
CREATE UNIQUE INDEX "LlmModelMigration_active_source_key" ON "LlmModelMigration"("sourceModelSlug") WHERE "isReverted" = false;

-- AddForeignKey
ALTER TABLE "LlmModel" ADD CONSTRAINT "LlmModel_providerId_fkey" FOREIGN KEY ("providerId") REFERENCES "LlmProvider"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LlmModel" ADD CONSTRAINT "LlmModel_creatorId_fkey" FOREIGN KEY ("creatorId") REFERENCES "LlmModelCreator"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LlmModelCost" ADD CONSTRAINT "LlmModelCost_llmModelId_fkey" FOREIGN KEY ("llmModelId") REFERENCES "LlmModel"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddCheckConstraints (enforce data integrity)
ALTER TABLE "LlmModel"
    ADD CONSTRAINT "LlmModel_priceTier_check" CHECK ("priceTier" BETWEEN 1 AND 3);

ALTER TABLE "LlmModelCost"
    ADD CONSTRAINT "LlmModelCost_creditCost_check" CHECK ("creditCost" >= 0);

ALTER TABLE "LlmModelMigration"
    ADD CONSTRAINT "LlmModelMigration_nodeCount_check" CHECK ("nodeCount" >= 0),
    ADD CONSTRAINT "LlmModelMigration_customCreditCost_check" CHECK ("customCreditCost" IS NULL OR "customCreditCost" >= 0);
