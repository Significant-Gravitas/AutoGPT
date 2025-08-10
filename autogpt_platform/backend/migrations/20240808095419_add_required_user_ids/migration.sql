-- Update existing entries with NULL userId
UPDATE "AgentGraph"                  SET "userId" = '3e53486c-cf57-477e-ba2a-cb02dc828e1a' WHERE "userId" IS NULL;
UPDATE "AgentGraphExecution"         SET "userId" = '3e53486c-cf57-477e-ba2a-cb02dc828e1a' WHERE "userId" IS NULL;
UPDATE "AgentGraphExecutionSchedule" SET "userId" = '3e53486c-cf57-477e-ba2a-cb02dc828e1a' WHERE "userId" IS NULL;

-- AlterTable
ALTER TABLE "AgentGraph" ALTER COLUMN "userId" SET NOT NULL;

-- AlterTable
ALTER TABLE "AgentGraphExecution" ALTER COLUMN "userId" SET NOT NULL;

-- AlterTable
ALTER TABLE "AgentGraphExecutionSchedule" ALTER COLUMN "userId" SET NOT NULL;

-- AlterForeignKey
ALTER TABLE "AgentGraph" DROP CONSTRAINT "AgentGraph_userId_fkey";
ALTER TABLE "AgentGraph" ADD CONSTRAINT "AgentGraph_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AlterForeignKey
ALTER TABLE "AgentGraphExecution" DROP CONSTRAINT "AgentGraphExecution_userId_fkey";
ALTER TABLE "AgentGraphExecution" ADD CONSTRAINT "AgentGraphExecution_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AlterForeignKey
ALTER TABLE "AgentGraphExecutionSchedule" DROP CONSTRAINT "AgentGraphExecutionSchedule_userId_fkey";
ALTER TABLE "AgentGraphExecutionSchedule" ADD CONSTRAINT "AgentGraphExecutionSchedule_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
