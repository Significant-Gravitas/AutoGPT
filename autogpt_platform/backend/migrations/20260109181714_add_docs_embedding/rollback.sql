-- Rollback migration: 20260109181714_add_docs_embedding
-- This SQL file reverses the add_docs_embedding migration
-- Run this manually when you need to rollback the migration

-- Drop indexes first (must drop before dropping the table)
DROP INDEX IF EXISTS "UnifiedContentEmbedding_embedding_idx";
DROP INDEX IF EXISTS "UnifiedContentEmbedding_contentType_contentId_userId_key";
DROP INDEX IF EXISTS "UnifiedContentEmbedding_contentType_userId_idx";
DROP INDEX IF EXISTS "UnifiedContentEmbedding_userId_idx";
DROP INDEX IF EXISTS "UnifiedContentEmbedding_contentType_idx";

-- Drop the table
DROP TABLE IF EXISTS "UnifiedContentEmbedding";

-- Drop the enum type
DROP TYPE IF EXISTS "ContentType";

-- NOTE: Do NOT drop the pgvector extension as it may be used by other features
-- If you need to drop it, uncomment the line below:
-- DROP EXTENSION IF EXISTS vector;
