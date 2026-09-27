-- The dream pass reaper lists the open passes whose lease lapsed, oldest
-- first, across every user (backend/copilot/dream/reaper.py). This index
-- keeps that a range scan per open status. A new index on an existing table;
-- no row is rewritten.

-- CreateIndex
CREATE INDEX "DreamPass_status_leaseExpiresAt_idx" ON "DreamPass"("status", "leaseExpiresAt");
