CREATE INDEX IF NOT EXISTS "ExpertWorkflow_expertId_createdAt_id_idx" ON "ExpertWorkflow"("expertId", "createdAt", "id");
DROP INDEX IF EXISTS "ExpertWorkflow_expertId_idx";
