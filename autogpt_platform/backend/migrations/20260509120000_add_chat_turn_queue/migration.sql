-- Session-level lifecycle for the per-user soft running cap +
-- cross-session queue.  ``idle`` (DEFAULT, no turn in flight),
-- ``queued`` (waiting for a running slot), ``running`` (a turn is
-- being processed).  A Postgres enum is used (rather than TEXT) so
-- typos and invalid values are rejected at the DB layer; adding a
-- future state needs an ``ALTER TYPE ... ADD VALUE`` migration, which
-- is cheap on PG 12+.
CREATE TYPE "ChatSessionStatus" AS ENUM ('idle', 'queued', 'running');

ALTER TABLE "ChatSession"
    ADD COLUMN "chatStatus" "ChatSessionStatus" NOT NULL DEFAULT 'idle';

-- Single compound index covers all three ChatSession queries:
--   * cap-count (count by userId + chatStatus)
--   * queue-list (find_many WHERE userId + chatStatus ORDER BY updatedAt)
--   * sidebar list (find_many WHERE userId ORDER BY updatedAt desc) —
--     Postgres scans the per-userId index range and sorts in memory.
-- Drop the prior (userId, updatedAt) index so the 3-col one is the
-- only path for this table's per-user queries.
DROP INDEX IF EXISTS "ChatSession_userId_updatedAt_idx";

CREATE INDEX "ChatSession_user_status_idx"
    ON "ChatSession" ("userId", "chatStatus", "updatedAt");

-- ChatMessage carries an optional per-row JSONB metadata bag for the
-- dispatcher's submit-time payload on the user row that triggered a
-- queued turn (file_ids, mode, model, permissions, context,
-- request_arrival_at).  Cleared / unused on every history row.
ALTER TABLE "ChatMessage"
    ADD COLUMN "metadata" JSONB;
