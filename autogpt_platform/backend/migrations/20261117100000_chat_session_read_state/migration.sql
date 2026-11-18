-- The sidebar's unread and error markers come from the server, so every
-- browser and device shows the same ones (SECRT-3542). All three start null:
-- chats whose turns ended before this count as read.

-- Nullable columns: a catalog-only change, no table rewrite.
-- AlterTable
ALTER TABLE "ChatSession" ADD COLUMN "lastTurnEndedAt" TIMESTAMP(3),
ADD COLUMN "lastTurnFailed" BOOLEAN,
ADD COLUMN "lastReadAt" TIMESTAMP(3);
