/*
  Warnings:

  - You are about to drop the column `search` on the `StoreListingVersion` table. All the data in the column will be lost.

*/
-- DropIndex
DROP INDEX "UnifiedContentEmbedding_search_idx";

-- AlterTable
ALTER TABLE "StoreListingVersion" DROP COLUMN "search";

-- CreateTable
CREATE TABLE "MemoryEpisodeLog" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT NOT NULL,
    "sessionId" TEXT,
    "groupId" TEXT NOT NULL,
    "episodeName" TEXT NOT NULL,
    "episodeBody" TEXT NOT NULL,
    "source" TEXT NOT NULL,
    "sourceDescription" TEXT,

    CONSTRAINT "MemoryEpisodeLog_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "MemoryEpisodeLog_userId_idx" ON "MemoryEpisodeLog"("userId");

-- CreateIndex
CREATE INDEX "MemoryEpisodeLog_createdAt_idx" ON "MemoryEpisodeLog"("createdAt");

-- AddForeignKey
ALTER TABLE "MemoryEpisodeLog" ADD CONSTRAINT "MemoryEpisodeLog_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
