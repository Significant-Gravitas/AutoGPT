-- Add durationMs column to ChatMessage for persisting turn elapsed time.
ALTER TABLE "ChatMessage" ADD COLUMN "durationMs" INTEGER;
