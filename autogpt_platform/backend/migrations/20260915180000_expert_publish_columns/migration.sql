-- AlterTable
ALTER TABLE "Expert" ADD COLUMN     "publishedFromExpertId" TEXT,
ADD COLUMN     "publishedPackage" BYTEA;

-- CreateIndex
CREATE UNIQUE INDEX "Expert_publishedFromExpertId_key" ON "Expert"("publishedFromExpertId");

-- AddForeignKey
ALTER TABLE "Expert" ADD CONSTRAINT "Expert_publishedFromExpertId_fkey" FOREIGN KEY ("publishedFromExpertId") REFERENCES "Expert"("id") ON DELETE SET NULL ON UPDATE CASCADE;
