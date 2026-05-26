-- Add WORKSPACE_FILE and CHAT_SESSION to the ContentType enum and extend the
-- defense-in-depth CHECK constraint so these per-user types can never be
-- inserted with a NULL userId (which would otherwise leak across users via
-- the public branch of hybrid_search's user filter).

ALTER TYPE "ContentType" ADD VALUE IF NOT EXISTS 'WORKSPACE_FILE';
ALTER TYPE "ContentType" ADD VALUE IF NOT EXISTS 'CHAT_SESSION';

-- Sweep any stray null-userId rows first (none expected) so the constraint
-- swap doesn't fail in environments where rows were hand-inserted.
DELETE FROM "UnifiedContentEmbedding"
WHERE "contentType" IN ('WORKSPACE_FILE', 'CHAT_SESSION')
  AND "userId" IS NULL;

-- Replace the LIBRARY_AGENT-only CHECK with one that covers all three
-- per-user content types. Drop-then-add keeps idempotency on re-runs.
ALTER TABLE "UnifiedContentEmbedding"
DROP CONSTRAINT IF EXISTS "UnifiedContentEmbedding_library_agent_requires_user";

ALTER TABLE "UnifiedContentEmbedding"
DROP CONSTRAINT IF EXISTS "UnifiedContentEmbedding_user_scoped_requires_user";

ALTER TABLE "UnifiedContentEmbedding"
ADD CONSTRAINT "UnifiedContentEmbedding_user_scoped_requires_user"
CHECK (
  "contentType" NOT IN ('LIBRARY_AGENT', 'WORKSPACE_FILE', 'CHAT_SESSION')
  OR "userId" IS NOT NULL
);
