-- AlterTable
ALTER TABLE "LibraryAgent" ADD COLUMN     "storeListingVersionId" TEXT;

-- CreateIndex
CREATE INDEX "LibraryAgent_storeListingVersionId_idx" ON "LibraryAgent"("storeListingVersionId");

-- AddForeignKey
ALTER TABLE "LibraryAgent" ADD CONSTRAINT "LibraryAgent_storeListingVersionId_fkey" FOREIGN KEY ("storeListingVersionId") REFERENCES "StoreListingVersion"("id") ON DELETE SET NULL ON UPDATE CASCADE;
