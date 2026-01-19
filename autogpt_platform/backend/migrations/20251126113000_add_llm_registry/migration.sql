-- CreateEnum
CREATE TYPE "LlmCostUnit" AS ENUM ('RUN', 'TOKENS');

-- CreateTable
CREATE TABLE "LlmProvider" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "name" TEXT NOT NULL,
    "displayName" TEXT NOT NULL,
    "description" TEXT,
    "defaultCredentialProvider" TEXT,
    "defaultCredentialId" TEXT,
    "defaultCredentialType" TEXT,
    "supportsTools" BOOLEAN NOT NULL DEFAULT TRUE,
    "supportsJsonOutput" BOOLEAN NOT NULL DEFAULT TRUE,
    "supportsReasoning" BOOLEAN NOT NULL DEFAULT FALSE,
    "supportsParallelTool" BOOLEAN NOT NULL DEFAULT FALSE,
    "metadata" JSONB NOT NULL DEFAULT '{}'::jsonb,

    CONSTRAINT "LlmProvider_pkey" PRIMARY KEY ("id"),
    CONSTRAINT "LlmProvider_name_key" UNIQUE ("name")
);

-- CreateTable
CREATE TABLE "LlmModel" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "slug" TEXT NOT NULL,
    "displayName" TEXT NOT NULL,
    "description" TEXT,
    "providerId" TEXT NOT NULL,
    "contextWindow" INTEGER NOT NULL,
    "maxOutputTokens" INTEGER,
    "isEnabled" BOOLEAN NOT NULL DEFAULT TRUE,
    "capabilities" JSONB NOT NULL DEFAULT '{}'::jsonb,
    "metadata" JSONB NOT NULL DEFAULT '{}'::jsonb,

    CONSTRAINT "LlmModel_pkey" PRIMARY KEY ("id"),
    CONSTRAINT "LlmModel_slug_key" UNIQUE ("slug")
);

-- CreateTable
CREATE TABLE "LlmModelCost" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "unit" "LlmCostUnit" NOT NULL DEFAULT 'RUN',
    "creditCost" INTEGER NOT NULL,
    "credentialProvider" TEXT NOT NULL,
    "credentialId" TEXT,
    "credentialType" TEXT,
    "currency" TEXT,
    "metadata" JSONB NOT NULL DEFAULT '{}'::jsonb,
    "llmModelId" TEXT NOT NULL,

    CONSTRAINT "LlmModelCost_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "LlmModel_providerId_isEnabled_idx" ON "LlmModel"("providerId", "isEnabled");

-- CreateIndex
CREATE INDEX "LlmModel_slug_idx" ON "LlmModel"("slug");

-- CreateIndex
CREATE INDEX "LlmModelCost_llmModelId_idx" ON "LlmModelCost"("llmModelId");

-- CreateIndex
CREATE INDEX "LlmModelCost_credentialProvider_idx" ON "LlmModelCost"("credentialProvider");

-- CreateIndex
CREATE UNIQUE INDEX "LlmModelCost_llmModelId_credentialProvider_unit_key" ON "LlmModelCost"("llmModelId", "credentialProvider", "unit");

-- AddForeignKey
ALTER TABLE "LlmModel" ADD CONSTRAINT "LlmModel_providerId_fkey" FOREIGN KEY ("providerId") REFERENCES "LlmProvider"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LlmModelCost" ADD CONSTRAINT "LlmModelCost_llmModelId_fkey" FOREIGN KEY ("llmModelId") REFERENCES "LlmModel"("id") ON DELETE CASCADE ON UPDATE CASCADE;

