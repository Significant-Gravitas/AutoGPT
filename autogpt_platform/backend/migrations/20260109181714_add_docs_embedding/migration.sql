-- CreateExtension
-- Supabase: pgvector must be enabled via Dashboard → Database → Extensions first
-- Creates extension in current schema (determined by search_path from DATABASE_URL ?schema= param)
-- This ensures vector type is in the same schema as tables, making ::vector work without explicit qualification
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS "vector";
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'vector extension not available or already exists, skipping';
END $$;

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
-- NULLS NOT DISTINCT ensures only one public (NULL userId) embedding per contentType+contentId
-- Requires PostgreSQL 15+. Supabase uses PostgreSQL 15+.
CREATE UNIQUE INDEX "UnifiedContentEmbedding_contentType_contentId_userId_key" ON "UnifiedContentEmbedding"("contentType", "contentId", "userId") NULLS NOT DISTINCT;

-- CreateIndex
-- HNSW index for fast vector similarity search on embeddings
-- Uses cosine distance operator (<=>), which matches the query in hybrid_search.py
-- Note: Drop first in case Prisma created a btree index (Prisma doesn't support HNSW)
DROP INDEX IF EXISTS "UnifiedContentEmbedding_embedding_idx";
CREATE INDEX "UnifiedContentEmbedding_embedding_idx" ON "UnifiedContentEmbedding" USING hnsw ("embedding" vector_cosine_ops);
