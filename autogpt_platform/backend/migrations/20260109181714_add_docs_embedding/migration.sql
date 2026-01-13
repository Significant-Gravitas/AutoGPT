-- CreateExtension
-- Create pgvector extension in the current schema (determined by DATABASE_URL schema parameter)
-- This works with both "public" (CI/default) and "platform" (production) schemas
CREATE EXTENSION IF NOT EXISTS "vector";

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
