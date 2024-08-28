-- DropForeignKey
ALTER TABLE "AgentGraphExecutionSchedule" DROP CONSTRAINT "AgentGraphExecutionSchedule_userId_fkey";

-- AlterTable
ALTER TABLE "AgentGraphExecutionSchedule" ALTER COLUMN "userId" DROP NOT NULL;

-- AddForeignKey
ALTER TABLE "AgentGraphExecutionSchedule" ADD CONSTRAINT "AgentGraphExecutionSchedule_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE SET NULL ON UPDATE CASCADE;
