CREATE TABLE "LlmProviderSettings" (
    "id" TEXT NOT NULL DEFAULT 'default',
    "enabled" BOOLEAN NOT NULL DEFAULT false,
    "provider" TEXT NOT NULL,
    "useLocal" BOOLEAN NOT NULL DEFAULT false,
    "baseUrl" TEXT,
    "encryptedApiKey" TEXT,
    "model" TEXT NOT NULL,
    "titleModel" TEXT NOT NULL,
    "fastStandardModel" TEXT NOT NULL,
    "fastAdvancedModel" TEXT NOT NULL,
    "thinkingStandardModel" TEXT NOT NULL,
    "thinkingAdvancedModel" TEXT NOT NULL,
    "claudeAgentFallbackModel" TEXT NOT NULL,
    "requestTimeoutS" DOUBLE PRECISION NOT NULL DEFAULT 20,
    "maxRetries" INTEGER NOT NULL DEFAULT 1,
    "localRequestTimeoutS" DOUBLE PRECISION NOT NULL DEFAULT 1800,
    "storeEmbeddingModel" TEXT NOT NULL DEFAULT 'text-embedding-3-small',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LlmProviderSettings_pkey" PRIMARY KEY ("id")
);
