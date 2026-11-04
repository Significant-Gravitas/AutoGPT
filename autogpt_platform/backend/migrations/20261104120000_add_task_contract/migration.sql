-- The task contract: one structured record of what was asked, what done
-- looks like, the plan, the proof, the result, the open question and why it
-- is stuck. NULL for tasks that predate it; the API derives one for them.
ALTER TABLE "Task" ADD COLUMN "contract" JSONB;
ALTER TABLE "Task" ADD COLUMN "contractVersion" INTEGER NOT NULL DEFAULT 0;
