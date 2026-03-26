-- AlterTable
ALTER TABLE "platform"."User" ADD COLUMN "rateLimitTier" TEXT NOT NULL DEFAULT 'standard';
