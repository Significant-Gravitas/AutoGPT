-- These changes are part of improvements to our API key system.
-- See https://github.com/Significant-Gravitas/AutoGPT/pull/10796 for context.

-- Add 'salt' column for Scrypt hashing
ALTER TABLE "APIKey" ADD COLUMN     "salt" TEXT;

-- Rename columns for clarity
ALTER TABLE "APIKey" RENAME COLUMN "key" TO "hash";
ALTER TABLE "APIKey" RENAME COLUMN "prefix" TO "head";
ALTER TABLE "APIKey" RENAME COLUMN "postfix" TO "tail";
