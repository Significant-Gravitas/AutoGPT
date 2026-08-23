-- CreateEnum
-- Lowercase members mirror the Prisma enum (same convention as
-- "ChatSessionStatus"), so the generated client and the column agree.
DO $$
BEGIN
    CREATE TYPE "TriggerSource" AS ENUM ('cron', 'webhook', 'manual', 'delegated');
EXCEPTION
    WHEN duplicate_object THEN NULL;
END
$$;

-- AlterTable
-- ADD COLUMN takes ACCESS EXCLUSIVE on AgentGraphExecution. A NOT NULL
-- column with a *constant* default is a catalog-only change on PG 11+ (no
-- table rewrite), so the lock window stays as brief as the nullable
-- "expertId" add in 20260804000000. IF NOT EXISTS keeps a rerun — or an
-- operator who added the column out-of-band — from failing the deploy.
--
-- Existing rows therefore back-fill to 'manual' for free, via the default.
-- That is the deliberate choice for history: the only other value we could
-- infer is 'delegated' for rows with a parentGraphExecutionId, and a
-- full-table UPDATE on AgentGraphExecution is exactly the long-running
-- write this table's migrations avoid. 'manual' reads as "user-initiated /
-- unknown provenance", which is the honest answer for pre-cutover rows —
-- and the proactive watchers only ever read rows created after this ships.
ALTER TABLE "AgentGraphExecution"
ADD COLUMN IF NOT EXISTS "triggerSource" "TriggerSource" NOT NULL DEFAULT 'manual';
