-- CreateExtension in platform schema (avoids search_path issues with Prisma)
-- Note: Platform schema is configured via DATABASE_URL schema parameter
CREATE EXTENSION IF NOT EXISTS "vector" WITH SCHEMA "platform";

-- CreateEnum
CREATE TYPE "ContentType" AS ENUM ('STORE_AGENT', 'BLOCK', 'INTEGRATION', 'DOCUMENTATION', 'LIBRARY_AGENT');

-- CreateTable
CREATE TABLE "UnifiedContentEmbedding" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "contentType" "ContentType" NOT NULL,
    "contentId" TEXT NOT NULL,
    "userId" TEXT,
    "embedding" vector(1536) NOT NULL,
    "searchableText" TEXT NOT NULL,
    "metadata" JSONB NOT NULL DEFAULT '{}',

    CONSTRAINT "UnifiedContentEmbedding_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "UnifiedContentEmbedding_contentType_idx" ON "UnifiedContentEmbedding"("contentType");

-- CreateIndex
CREATE INDEX "UnifiedContentEmbedding_userId_idx" ON "UnifiedContentEmbedding"("userId");

-- CreateIndex
CREATE INDEX "UnifiedContentEmbedding_contentType_userId_idx" ON "UnifiedContentEmbedding"("contentType", "userId");

-- CreateIndex
CREATE UNIQUE INDEX "UnifiedContentEmbedding_contentType_contentId_userId_key" ON "UnifiedContentEmbedding"("contentType", "contentId", "userId");
