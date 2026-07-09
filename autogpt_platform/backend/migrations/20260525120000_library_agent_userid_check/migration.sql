-- Defense-in-depth: LIBRARY_AGENT embeddings are always per-user. The
-- runtime filter in hybrid_search excludes NULL-userId LIBRARY_AGENT rows
-- to prevent cross-user leaks, but that's a single line of defense. This
-- CHECK constraint enforces the invariant at the storage layer so a future
-- write-path or migration bug can't introduce a leaking row.

-- Sweep any stray rows first (none expected — LibraryAgentHandler always
-- writes with userId, schedule_library_agent_embedding does too) so the
-- ALTER doesn't fail loudly in environments where someone hand-inserted.
DELETE FROM "UnifiedContentEmbedding"
WHERE "contentType" = 'LIBRARY_AGENT' AND "userId" IS NULL;

ALTER TABLE "UnifiedContentEmbedding"
ADD CONSTRAINT "UnifiedContentEmbedding_library_agent_requires_user"
CHECK ("contentType" != 'LIBRARY_AGENT' OR "userId" IS NOT NULL);
