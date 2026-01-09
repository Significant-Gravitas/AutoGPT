-- APScheduler tables (managed by APScheduler at runtime, baseline for Prisma)
CREATE TABLE IF NOT EXISTS "apscheduler_jobs" (
    "id" VARCHAR(191) NOT NULL,
    "next_run_time" DOUBLE PRECISION,
    "job_state" BYTEA NOT NULL,

    CONSTRAINT "apscheduler_jobs_pkey" PRIMARY KEY ("id")
);

CREATE TABLE IF NOT EXISTS "apscheduler_jobs_batched_notifications" (
    "id" VARCHAR(191) NOT NULL,
    "next_run_time" DOUBLE PRECISION,
    "job_state" BYTEA NOT NULL,

    CONSTRAINT "apscheduler_jobs_batched_notifications_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "ix_platform_apscheduler_jobs_next_run_time" ON "apscheduler_jobs"("next_run_time");
CREATE INDEX IF NOT EXISTS "ix_platform_apscheduler_jobs_batched_notifications_next_0b54" ON "apscheduler_jobs_batched_notifications"("next_run_time");
