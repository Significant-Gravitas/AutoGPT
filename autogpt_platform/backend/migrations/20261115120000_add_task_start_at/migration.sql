-- A task that starts by itself at a set time.

-- AlterTable
ALTER TABLE "Task" ADD COLUMN     "startAt" TIMESTAMP(3);

-- CreateIndex
CREATE INDEX "Task_status_startAt_idx" ON "Task"("status", "startAt");
