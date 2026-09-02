-- REL-005: durable scheduler occurrence with unique (scheduleId, fireTime)
CREATE TABLE IF NOT EXISTS "ScheduleOccurrence" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "scheduleId" TEXT NOT NULL,
    "fireTime" TIMESTAMP(3) NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'claimed',
    "executionId" TEXT UNIQUE,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS "ScheduleOccurrence_scheduleId_fireTime_key" ON "ScheduleOccurrence"("scheduleId", "fireTime");
CREATE INDEX IF NOT EXISTS "ScheduleOccurrence_scheduleId_idx" ON "ScheduleOccurrence"("scheduleId");
CREATE INDEX IF NOT EXISTS "ScheduleOccurrence_fireTime_idx" ON "ScheduleOccurrence"("fireTime");
