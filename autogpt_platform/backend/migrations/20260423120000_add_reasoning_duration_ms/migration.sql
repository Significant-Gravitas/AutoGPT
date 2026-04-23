-- Add reasoningDurationMs column to ChatMessage so the copilot UI can show
-- actual model-reasoning time instead of whole-turn wall clock (which
-- includes tool execution).
ALTER TABLE "ChatMessage" ADD COLUMN "reasoningDurationMs" INTEGER;
