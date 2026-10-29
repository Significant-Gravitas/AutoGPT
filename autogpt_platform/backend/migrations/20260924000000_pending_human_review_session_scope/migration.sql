-- Chat reviews were stored under a synthetic graph execution
-- ("copilot-session-<session id>"); they now carry the chat's session id.
-- Additive only: this runs before the new pods roll out, and the old model
-- reads graphExecId/graphId/graphVersion as non-null, so existing rows keep
-- their legacy values alongside the new chatSessionId.
ALTER TABLE "PendingHumanReview"
    ALTER COLUMN "graphExecId" DROP NOT NULL,
    ALTER COLUMN "graphId" DROP NOT NULL,
    ALTER COLUMN "graphVersion" DROP NOT NULL,
    ADD COLUMN "chatSessionId" TEXT;

UPDATE "PendingHumanReview"
SET "chatSessionId" = substr("graphExecId", length('copilot-session-') + 1)
WHERE "graphExecId" LIKE 'copilot-session-%';

-- A review is a graph execution's or a chat's; a legacy chat row keeps its
-- synthetic graph id beside the session until a cleanup migration clears it.
-- NOT VALID keeps the existing-row scan out of this migration's lock; the
-- next migration validates it.
ALTER TABLE "PendingHumanReview"
    ADD CONSTRAINT "PendingHumanReview_graph_or_chat_session_check"
    CHECK (
        ("graphExecId" IS NULL) <> ("chatSessionId" IS NULL)
        OR COALESCE("graphExecId" LIKE 'copilot-session-%', false)
    ) NOT VALID;

CREATE INDEX "PendingHumanReview_chatSessionId_status_idx" ON "PendingHumanReview"("chatSessionId", "status");
