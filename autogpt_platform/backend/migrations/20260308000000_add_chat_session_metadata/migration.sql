-- AlterTable: add metadata column to ChatSession
ALTER TABLE "ChatSession" ADD COLUMN "metadata" JSONB NOT NULL DEFAULT '{}';
