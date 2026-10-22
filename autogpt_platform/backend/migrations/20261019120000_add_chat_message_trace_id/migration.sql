-- The PostHog trace of the turn that wrote each assistant or reasoning row,
-- so a rating of the reply can name the trace in PostHog LLM analytics.

-- Nullable with no default: a catalog-only change, no table rewrite.
-- AlterTable
ALTER TABLE "ChatMessage" ADD COLUMN     "traceId" TEXT;
