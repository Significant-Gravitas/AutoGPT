-- CreateSchema (idempotent - safe if already exists)
CREATE SCHEMA IF NOT EXISTS "platform";

-- CreateEnum
CREATE TYPE "platform"."LlmCostUnit" AS ENUM ('RUN', 'TOKENS');

-- CreateTable
CREATE TABLE "platform"."LlmProvider" (
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
CREATE TABLE "platform"."LlmModelCreator" (
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
CREATE TABLE "platform"."LlmModel" (
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
CREATE TABLE "platform"."LlmModelCost" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "unit" "platform"."LlmCostUnit" NOT NULL DEFAULT 'RUN',
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
CREATE TABLE "platform"."LlmModelMigration" (
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
CREATE UNIQUE INDEX "LlmProvider_name_key" ON "platform"."LlmProvider"("name");

-- CreateIndex
CREATE UNIQUE INDEX "LlmModelCreator_name_key" ON "platform"."LlmModelCreator"("name");

-- CreateIndex
CREATE UNIQUE INDEX "LlmModel_slug_key" ON "platform"."LlmModel"("slug");

-- CreateIndex
CREATE INDEX "LlmModel_providerId_isEnabled_idx" ON "platform"."LlmModel"("providerId", "isEnabled");

-- CreateIndex
CREATE INDEX "LlmModel_creatorId_idx" ON "platform"."LlmModel"("creatorId");

-- CreateIndex
CREATE INDEX "LlmModelCost_credentialProvider_idx" ON "platform"."LlmModelCost"("credentialProvider");

-- CreateIndex (partial unique for default costs - no specific credential)
CREATE UNIQUE INDEX "LlmModelCost_default_cost_key" ON "platform"."LlmModelCost"("llmModelId", "credentialProvider", "unit") WHERE "credentialId" IS NULL;

-- CreateIndex (partial unique for credential-specific costs)
CREATE UNIQUE INDEX "LlmModelCost_credential_cost_key" ON "platform"."LlmModelCost"("llmModelId", "credentialProvider", "credentialId", "unit") WHERE "credentialId" IS NOT NULL;

-- CreateIndex
CREATE INDEX "LlmModelMigration_targetModelSlug_idx" ON "platform"."LlmModelMigration"("targetModelSlug");

-- CreateIndex
CREATE INDEX "LlmModelMigration_isReverted_idx" ON "platform"."LlmModelMigration"("isReverted");

-- CreateIndex
CREATE INDEX "LlmModelMigration_sourceModelSlug_isReverted_idx" ON "platform"."LlmModelMigration"("sourceModelSlug", "isReverted");

-- CreateIndex (partial unique to prevent multiple active migrations per source)
CREATE UNIQUE INDEX "LlmModelMigration_active_source_key" ON "platform"."LlmModelMigration"("sourceModelSlug") WHERE "isReverted" = false;

-- AddForeignKey
ALTER TABLE "platform"."LlmModel" ADD CONSTRAINT "LlmModel_providerId_fkey" FOREIGN KEY ("providerId") REFERENCES "platform"."LlmProvider"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "platform"."LlmModel" ADD CONSTRAINT "LlmModel_creatorId_fkey" FOREIGN KEY ("creatorId") REFERENCES "platform"."LlmModelCreator"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "platform"."LlmModelCost" ADD CONSTRAINT "LlmModelCost_llmModelId_fkey" FOREIGN KEY ("llmModelId") REFERENCES "platform"."LlmModel"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddCheckConstraints (enforce data integrity)
ALTER TABLE "platform"."LlmModel"
    ADD CONSTRAINT "LlmModel_priceTier_check" CHECK ("priceTier" BETWEEN 1 AND 3);

ALTER TABLE "platform"."LlmModelCost"
    ADD CONSTRAINT "LlmModelCost_creditCost_check" CHECK ("creditCost" >= 0);

ALTER TABLE "platform"."LlmModelMigration"
    ADD CONSTRAINT "LlmModelMigration_nodeCount_check" CHECK ("nodeCount" >= 0),
    ADD CONSTRAINT "LlmModelMigration_customCreditCost_check" CHECK ("customCreditCost" IS NULL OR "customCreditCost" >= 0);
