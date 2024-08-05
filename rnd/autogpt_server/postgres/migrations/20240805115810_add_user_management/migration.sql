-- AlterTable
ALTER TABLE "AgentGraph" ADD COLUMN     "userId" TEXT;

-- AlterTable
ALTER TABLE "AgentGraphExecution" ADD COLUMN     "userId" TEXT;

-- AlterTable
ALTER TABLE "AgentGraphExecutionSchedule" ADD COLUMN     "userId" TEXT;

-- CreateTable
CREATE TABLE "User" (
    "id" TEXT NOT NULL,
    "email" TEXT NOT NULL,
    "name" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "User_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "User_email_key" ON "User"("email");

-- AddForeignKey
ALTER TABLE "AgentGraph" ADD CONSTRAINT "AgentGraph_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AgentGraphExecution" ADD CONSTRAINT "AgentGraphExecution_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AgentGraphExecutionSchedule" ADD CONSTRAINT "AgentGraphExecutionSchedule_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE SET NULL ON UPDATE CASCADE;
