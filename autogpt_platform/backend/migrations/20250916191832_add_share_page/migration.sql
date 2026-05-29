/*
  Warnings:

  - A unique constraint covering the columns `[shareToken]` on the table `AgentGraphExecution` will be added. If there are existing duplicate values, this will fail.

*/
-- AlterTable
ALTER TABLE "AgentGraphExecution" ADD COLUMN     "isShared" BOOLEAN NOT NULL DEFAULT false,
ADD COLUMN     "shareToken" TEXT,
ADD COLUMN     "sharedAt" TIMESTAMP(3);

-- CreateIndex
CREATE UNIQUE INDEX "AgentGraphExecution_shareToken_key" ON "AgentGraphExecution"("shareToken");

-- CreateIndex
CREATE INDEX "AgentGraphExecution_shareToken_idx" ON "AgentGraphExecution"("shareToken");

-- RenameIndex
ALTER INDEX "APIKey_key_key" RENAME TO "APIKey_hash_key";

-- RenameIndex
ALTER INDEX "APIKey_prefix_name_idx" RENAME TO "APIKey_head_name_idx";
