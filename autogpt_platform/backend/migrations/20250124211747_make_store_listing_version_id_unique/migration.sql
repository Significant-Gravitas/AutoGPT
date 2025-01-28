/*
  Warnings:

  - A unique constraint covering the columns `[storeListingVersionId]` on the table `StoreListingSubmission` will be added. If there are existing duplicate values, this will fail.

*/
-- CreateIndex
CREATE UNIQUE INDEX "StoreListingSubmission_storeListingVersionId_key" ON "StoreListingSubmission"("storeListingVersionId");
