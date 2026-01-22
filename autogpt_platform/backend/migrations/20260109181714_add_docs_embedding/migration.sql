-- CreateExtension
-- Supabase: pgvector must be enabled via Dashboard → Database → Extensions first
-- Ensures vector extension is in the current schema (from DATABASE_URL ?schema= param)
-- If it exists in a different schema (e.g., public), we drop and recreate it in the current schema
-- This ensures vector type is in the same schema as tables, making ::vector work without explicit qualification
DO $$
DECLARE
    current_schema_name text;
    vector_schema text;
BEGIN
    -- Get the current schema from search_path
    SELECT current_schema() INTO current_schema_name;

    -- Check if vector extension exists and which schema it's in
    SELECT n.nspname INTO vector_schema
    FROM pg_extension e
    JOIN pg_namespace n ON e.extnamespace = n.oid
    WHERE e.extname = 'vector';

    -- Handle removal if in wrong schema
    IF vector_schema IS NOT NULL AND vector_schema != current_schema_name THEN
        BEGIN
            -- Vector exists in a different schema, drop it first
            RAISE WARNING 'pgvector found in schema "%" but need it in "%". Dropping and reinstalling...',
                vector_schema, current_schema_name;
            EXECUTE 'DROP EXTENSION IF EXISTS vector CASCADE';
        EXCEPTION WHEN OTHERS THEN
            RAISE EXCEPTION 'Failed to drop pgvector from schema "%": %. You may need to drop it manually.',
                vector_schema, SQLERRM;
        END;
    END IF;

    -- Create extension in current schema (let it fail naturally if not available)
    EXECUTE format('CREATE EXTENSION IF NOT EXISTS vector SCHEMA %I', current_schema_name);
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
