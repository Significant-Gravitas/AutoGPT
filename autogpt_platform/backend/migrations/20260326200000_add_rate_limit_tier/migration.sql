-- AlterTable
ALTER TABLE "User" ADD COLUMN "rateLimitTier" TEXT NOT NULL DEFAULT 'standard';
