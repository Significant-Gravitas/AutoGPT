-- AlterTable
-- Nullable with no default: Postgres records this in the catalog only, so
-- it does not rewrite the table. Existing rows read as NULL and fall back
-- to the session's own route.
ALTER TABLE "ChatMessage" ADD COLUMN     "llmAuthProvider" TEXT,
ADD COLUMN     "llmCredentialId" TEXT;
