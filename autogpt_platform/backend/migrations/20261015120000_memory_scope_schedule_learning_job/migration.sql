-- The account's skill-learning cron (copilot/dream/scheduling.py), recorded
-- like the scope's other crons. An expert's scope has none, so its row leaves
-- this null.
ALTER TABLE "MemoryScopeSchedule" ADD COLUMN "learningJobId" TEXT;
