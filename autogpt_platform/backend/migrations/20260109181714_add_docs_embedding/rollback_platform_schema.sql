-- Rollback migration: 20260109181714_add_docs_embedding (for platform schema)
-- This SQL file reverses the add_docs_embedding migration for Supabase with platform schema
-- Run this manually when you need to rollback the migration in the platform schema

-- Drop indexes first (must drop before dropping the table)
DROP INDEX IF EXISTS "platform"."UnifiedContentEmbedding_embedding_idx";
DROP INDEX IF EXISTS "platform"."UnifiedContentEmbedding_contentType_contentId_userId_key";
DROP INDEX IF EXISTS "platform"."UnifiedContentEmbedding_contentType_userId_idx";
DROP INDEX IF EXISTS "platform"."UnifiedContentEmbedding_userId_idx";
DROP INDEX IF EXISTS "platform"."UnifiedContentEmbedding_contentType_idx";

-- Drop the table
DROP TABLE IF EXISTS "platform"."UnifiedContentEmbedding";

-- Drop the enum type (in platform schema)
DROP TYPE IF EXISTS "platform"."ContentType";

-- NOTE: Do NOT drop the pgvector extension as it may be used by other features
-- The vector extension is installed at the database level, not per-schema
-- If you need to drop it, uncomment the line below:
-- DROP EXTENSION IF EXISTS vector;
