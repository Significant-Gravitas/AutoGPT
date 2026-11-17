-- Resetting an expert archives its chats (SECRT-2692): they stay in the
-- database but leave every chat list. Nullable with no backfill, so every
-- existing chat stays visible.
ALTER TABLE "ChatSession" ADD COLUMN "archivedAt" TIMESTAMP(3);
