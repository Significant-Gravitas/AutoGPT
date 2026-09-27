-- The dream pass reaper (backend/copilot/dream/reaper.py) lists, across every
-- user, the open passes whose lease lapsed and the closed passes whose cleanup
-- it has not finished, oldest first. cleanupPendingAt marks the second kind:
-- set when the reaper closes a row, cleared once it has cleaned up after the
-- pass. The two indexes keep both scans a range scan per status. A nullable
-- column and two new indexes on an existing table; no row is rewritten.

-- AlterTable
ALTER TABLE "DreamPass" ADD COLUMN     "cleanupPendingAt" TIMESTAMP(3);

-- CreateIndex
CREATE INDEX "DreamPass_status_leaseExpiresAt_idx" ON "DreamPass"("status", "leaseExpiresAt");

-- CreateIndex
CREATE INDEX "DreamPass_status_cleanupPendingAt_idx" ON "DreamPass"("status", "cleanupPendingAt");
