-- Step 2/2: tighten the per-user CHECK constraint on UnifiedContentEmbedding
-- to cover WORKSPACE_FILE and CHAT_SESSION (added in the preceding
-- migration). The split is required because Postgres errors with
-- "unsafe use of new value" if a CHECK / DELETE in the same transaction
-- as `ALTER TYPE ... ADD VALUE` references the new enum values.

-- Sweep any stray rows with NULL userId for the per-user types so the
-- new CHECK doesn't fail in environments where someone hand-inserted
-- (no rows are expected in practice).
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
