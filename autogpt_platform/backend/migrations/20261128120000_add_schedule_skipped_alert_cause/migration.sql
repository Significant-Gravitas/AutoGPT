-- A schedule whose runs fail the workflow's checks when they fire is kept,
-- each run is skipped, and its owner is told once (#17245).
ALTER TYPE "AlertCause" ADD VALUE 'SCHEDULE_SKIPPED';
