-- CreateTable
CREATE TABLE "ExpertPublishedPackage" (
    "expertId" TEXT NOT NULL,
    "package" BYTEA NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "ExpertPublishedPackage_pkey" PRIMARY KEY ("expertId")
);

-- AddForeignKey
ALTER TABLE "ExpertPublishedPackage" ADD CONSTRAINT "ExpertPublishedPackage_expertId_fkey" FOREIGN KEY ("expertId") REFERENCES "Expert"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- Move packages written under the column 20260915180000_expert_publish_columns
-- added, so a template published before this migration keeps serving what was
-- published.
INSERT INTO "ExpertPublishedPackage" ("expertId", "package")
SELECT "id", "publishedPackage" FROM "Expert" WHERE "publishedPackage" IS NOT NULL;

-- AlterTable
ALTER TABLE "Expert" DROP COLUMN "publishedPackage";
