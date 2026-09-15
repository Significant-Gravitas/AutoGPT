-- AlterTable
ALTER TABLE "SkillListingVersion" ADD COLUMN     "license" TEXT,
ADD COLUMN     "sourceRepo" TEXT,
ADD COLUMN     "sourceUrl" TEXT;

-- CreateTable
CREATE TABLE "SkillListingFile" (
    "id" TEXT NOT NULL,
    "skillListingVersionId" TEXT NOT NULL,
    "relativePath" TEXT NOT NULL,
    "content" BYTEA NOT NULL,
    "sizeBytes" INTEGER NOT NULL,
    "isExecutable" BOOLEAN NOT NULL DEFAULT false,

    CONSTRAINT "SkillListingFile_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "SkillListingFile_skillListingVersionId_relativePath_key" ON "SkillListingFile"("skillListingVersionId", "relativePath");

-- AddForeignKey
ALTER TABLE "SkillListingFile" ADD CONSTRAINT "SkillListingFile_skillListingVersionId_fkey" FOREIGN KEY ("skillListingVersionId") REFERENCES "SkillListingVersion"("id") ON DELETE CASCADE ON UPDATE CASCADE;
