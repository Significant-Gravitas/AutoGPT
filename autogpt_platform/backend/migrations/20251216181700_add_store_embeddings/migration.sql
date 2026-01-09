-- Migration: Add pgvector extension and StoreListingEmbedding table
-- This enables hybrid search combining semantic (embedding) and lexical (tsvector) search

-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table to store embeddings for store listing versions
CREATE TABLE "StoreListingEmbedding" (
    "id" TEXT NOT NULL DEFAULT gen_random_uuid(),
    "storeListingVersionId" TEXT NOT NULL,
    "embedding" public.vector(1536),  -- OpenAI text-embedding-3-small produces 1536 dimensions
    "searchableText" TEXT,     -- The text that was embedded (for debugging/recomputation)
    "contentHash" TEXT,        -- MD5 hash of searchable text for change detection
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "StoreListingEmbedding_pkey" PRIMARY KEY ("id")
);

-- Unique constraint: one embedding per listing version
CREATE UNIQUE INDEX "StoreListingEmbedding_storeListingVersionId_key"
    ON "StoreListingEmbedding"("storeListingVersionId");

-- HNSW index for fast approximate nearest neighbor search
-- Using cosine distance (vector_cosine_ops) which is standard for text embeddings
CREATE INDEX "StoreListingEmbedding_embedding_idx"
    ON "StoreListingEmbedding"
    USING hnsw ("embedding" public.vector_cosine_ops);

-- Index on content hash for fast lookup during change detection
CREATE INDEX "StoreListingEmbedding_contentHash_idx"
    ON "StoreListingEmbedding"("contentHash");

-- Foreign key to StoreListingVersion with CASCADE delete
-- When a listing version is deleted, its embedding is automatically removed
ALTER TABLE "StoreListingEmbedding"
    ADD CONSTRAINT "StoreListingEmbedding_storeListingVersionId_fkey"
    FOREIGN KEY ("storeListingVersionId")
    REFERENCES "StoreListingVersion"("id")
    ON DELETE CASCADE
    ON UPDATE CASCADE;
