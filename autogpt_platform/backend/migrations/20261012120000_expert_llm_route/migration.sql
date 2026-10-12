-- Per-expert AI connection: which chat LLM route (and credential) an expert's
-- new threads, routines, follow-ups and delegations run on. Both null means
-- the expert follows the owner's account default, which is every existing row.

-- Nullable with no default: a catalog-only change, no table rewrite.
-- AlterTable
ALTER TABLE "Expert" ADD COLUMN     "llmAuthProvider" TEXT,
ADD COLUMN     "llmCredentialId" TEXT;
