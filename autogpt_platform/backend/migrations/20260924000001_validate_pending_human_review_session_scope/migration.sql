-- Validate the check added NOT VALID in 20260924000000_pending_human_review_session_scope.
-- Its own transaction, so the scan holds only a SHARE UPDATE EXCLUSIVE lock
-- and review reads and writes proceed.
ALTER TABLE "PendingHumanReview" VALIDATE CONSTRAINT "PendingHumanReview_graph_or_session_check";
