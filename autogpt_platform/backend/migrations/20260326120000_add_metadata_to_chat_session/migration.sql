-- Add extensible metadata JSONB column to ChatSession.
-- New session-level flags (e.g. dry_run) live inside this JSON
-- so future additions need no extra migrations.
ALTER TABLE "ChatSession" ADD COLUMN "metadata" JSONB NOT NULL DEFAULT '{}';
