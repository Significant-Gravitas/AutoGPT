# Rollback Guide for add_docs_embedding Migration

This directory contains SQL files to rollback the `20260109181714_add_docs_embedding` migration.

## Files

- **`rollback.sql`**: For public schema (CI/local development)
- **`rollback_platform_schema.sql`**: For platform schema (Supabase production/dev)

## Usage

### Option 1: Via Supabase SQL Editor (Recommended for Cloud)

1. Go to your Supabase project dashboard
2. Navigate to **SQL Editor**
3. Copy the contents of the appropriate rollback file:
   - Use `rollback_platform_schema.sql` for dev/prod environments
   - Use `rollback.sql` for local testing
4. Run the SQL

### Option 2: Via psql (Local/CI)

```bash
# For public schema (local/CI)
psql $DATABASE_URL -f migrations/20260109181714_add_docs_embedding/rollback.sql

# For platform schema (Supabase)
psql $DATABASE_URL -f migrations/20260109181714_add_docs_embedding/rollback_platform_schema.sql
```

### Option 3: Via Prisma

Mark the migration as rolled back:

```bash
poetry run prisma migrate resolve --rolled-back 20260109181714_add_docs_embedding
```

Then run the rollback SQL manually via one of the methods above.

## What Gets Removed

1. ✅ HNSW vector similarity index
2. ✅ Unique constraint index (contentType + contentId + userId)
3. ✅ Performance indexes (contentType, userId, composite)
4. ✅ UnifiedContentEmbedding table
5. ✅ ContentType enum
6. ❌ pgvector extension (kept for safety - may be used elsewhere)

## After Rollback

If you want to reapply the migration:

```bash
poetry run prisma migrate deploy
```

This will reapply any pending migrations, including this one.

## Testing

To test the rollback in dev:

```bash
# 1. Run rollback SQL
psql $DEV_DATABASE_URL -f rollback_platform_schema.sql

# 2. Verify tables are gone
psql $DEV_DATABASE_URL -c "SELECT table_name FROM information_schema.tables WHERE table_schema = 'platform' AND table_name = 'UnifiedContentEmbedding';"

# 3. Reapply migration
poetry run prisma migrate deploy
```

## Caution

⚠️ **Data Loss Warning**: Rolling back this migration will permanently delete all stored embeddings. Make sure to backup if needed:

```sql
-- Backup embeddings before rollback
CREATE TABLE embeddings_backup AS
SELECT * FROM "platform"."UnifiedContentEmbedding";
```
