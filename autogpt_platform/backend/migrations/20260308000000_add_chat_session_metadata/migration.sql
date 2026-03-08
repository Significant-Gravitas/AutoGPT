-- AlterTable: add metadata column to ChatSession
ALTER TABLE "platform"."ChatSession" ADD COLUMN "metadata" JSONB NOT NULL DEFAULT '{}';
