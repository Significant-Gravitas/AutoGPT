-- Adds the global "auto-share executions" toggle to ChatSession.
-- When true, every run_agent execution in this chat (existing at
-- enable-time + any new ones while the share is active) is auto-
-- linked into the chat share via ChatLinkedShare rows.  Replaces
-- the previous per-execution checklist model.
ALTER TABLE "ChatSession"
ADD COLUMN "autoShareExecutions" BOOLEAN NOT NULL DEFAULT false;
