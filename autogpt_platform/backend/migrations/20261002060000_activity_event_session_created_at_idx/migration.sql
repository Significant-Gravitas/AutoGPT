-- Supports the per-turn <session_context> read
-- (data/activity_event.py::list_activity_events_by_type with session_id),
-- which filters on sessionId and orders by createdAt DESC. A bare sessionId
-- index makes Postgres fetch and sort every row a long-lived recurring
-- schedule has written to that session; the composite reads only the newest
-- window. Its leading column still serves every sessionId-only lookup, so it
-- replaces the bare index rather than sitting beside it.
--
-- Prisma wraps each migration in a transaction and Postgres rejects
-- CREATE INDEX CONCURRENTLY inside one. Deployments that can't tolerate the
-- brief write pause should build the CONCURRENTLY equivalent out-of-band
-- first; IF NOT EXISTS then makes this statement a no-op.

-- CreateIndex
CREATE INDEX IF NOT EXISTS "ActivityEvent_sessionId_createdAt_idx" ON "ActivityEvent"("sessionId", "createdAt");

-- DropIndex
DROP INDEX IF EXISTS "ActivityEvent_sessionId_idx";
