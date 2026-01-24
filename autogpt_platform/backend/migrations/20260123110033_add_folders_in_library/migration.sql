/*
  Warnings:

  - You are about to drop the column `search` on the `StoreListingVersion` table. All the data in the column will be lost.

*/
-- DropIndex
DROP INDEX "UnifiedContentEmbedding_search_idx";

-- AlterTable
ALTER TABLE "LibraryAgent" ADD COLUMN     "folderId" TEXT;

-- AlterTable
ALTER TABLE "StoreListingVersion" DROP COLUMN "search";

-- CreateTable
CREATE TABLE "LibraryFolder" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "icon" TEXT,
    "color" TEXT,
    "parentId" TEXT,
    "isDeleted" BOOLEAN NOT NULL DEFAULT false,

    CONSTRAINT "LibraryFolder_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "LibraryFolder_userId_isDeleted_idx" ON "LibraryFolder"("userId", "isDeleted");

-- CreateIndex
CREATE INDEX "LibraryFolder_parentId_idx" ON "LibraryFolder"("parentId");

-- CreateIndex
CREATE UNIQUE INDEX "LibraryFolder_userId_parentId_name_key" ON "LibraryFolder"("userId", "parentId", "name");

-- CreateIndex
CREATE INDEX "LibraryAgent_folderId_idx" ON "LibraryAgent"("folderId");

-- AddForeignKey
ALTER TABLE "LibraryAgent" ADD CONSTRAINT "LibraryAgent_folderId_fkey" FOREIGN KEY ("folderId") REFERENCES "LibraryFolder"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LibraryFolder" ADD CONSTRAINT "LibraryFolder_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LibraryFolder" ADD CONSTRAINT "LibraryFolder_parentId_fkey" FOREIGN KEY ("parentId") REFERENCES "LibraryFolder"("id") ON DELETE CASCADE ON UPDATE CASCADE;
