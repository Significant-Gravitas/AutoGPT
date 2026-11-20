-- CreateIndex
-- Supports the Home "Needs You" query
-- (copilot/db.py::get_sessions_with_pending_question), which filters on
-- jsonb_typeof(metadata->'pending_question') = 'object' and orders by
-- metadata->'pending_question'->>'asked_at' DESC. Without this index the
-- only usable index is (userId, chatStatus, updatedAt), so Postgres
-- heap-fetches and fully sorts every session the user owns on every Home
-- page load — the LIMIT only bounds the result, not the work.
--
-- Prisma's `migrate deploy` wraps each migration file in a single
-- transaction, and Postgres rejects `CREATE INDEX CONCURRENTLY` inside a
-- transaction block. The plain `CREATE INDEX` form briefly acquires a
-- ShareLock on ChatSession; for environments that can't tolerate that, run
-- the equivalent `CREATE INDEX CONCURRENTLY` out-of-band before this
-- migration ships and Postgres will skip the recreate via `IF NOT EXISTS`.
CREATE INDEX IF NOT EXISTS "ChatSession_pending_question_idx"
    ON "ChatSession" ("userId", (("metadata"->'pending_question'->>'asked_at')) DESC)
    WHERE jsonb_typeof("metadata"->'pending_question') = 'object';
