-- Add a pin flag to ChatSession so users can pin chats to the top of the
-- sidebar list. Defaults to false so existing sessions stay unpinned.
-- AlterTable
ALTER TABLE "ChatSession" ADD COLUMN "isPinned" BOOLEAN NOT NULL DEFAULT false;
