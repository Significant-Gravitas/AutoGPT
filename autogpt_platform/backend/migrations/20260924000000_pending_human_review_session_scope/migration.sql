-- Chat reviews were stored under a synthetic graph execution
-- ("copilot-session-<session id>"); they now carry the chat's session id and
-- no graph columns.
ALTER TABLE "PendingHumanReview"
    ALTER COLUMN "graphExecId" DROP NOT NULL,
    ALTER COLUMN "graphId" DROP NOT NULL,
    ALTER COLUMN "graphVersion" DROP NOT NULL,
    ADD COLUMN "sessionId" TEXT;

UPDATE "PendingHumanReview"
SET "sessionId" = substr("graphExecId", length('copilot-session-') + 1),
    "graphExecId" = NULL,
    "graphId" = NULL,
    "graphVersion" = NULL
WHERE "graphExecId" LIKE 'copilot-session-%';

-- Auto-approval records are keyed "auto_approve_<scope>_<node id>"; the
-- scope of a chat record is now the bare session id.
UPDATE "PendingHumanReview"
SET "nodeExecId" = 'auto_approve_' || substr("nodeExecId", length('auto_approve_copilot-session-') + 1)
WHERE "nodeExecId" LIKE 'auto\_approve\_copilot-session-%';

ALTER TABLE "PendingHumanReview"
    ADD CONSTRAINT "PendingHumanReview_graph_or_session_check"
    CHECK (("graphExecId" IS NULL) <> ("sessionId" IS NULL));

CREATE INDEX "PendingHumanReview_sessionId_status_idx" ON "PendingHumanReview"("sessionId", "status");
