-- CreateTable
CREATE TABLE "SkillListingFile" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "skillListingVersionId" TEXT NOT NULL,
    "relativePath" TEXT NOT NULL,
    "sizeBytes" INTEGER NOT NULL,
    "sha256" TEXT NOT NULL,
    "mimeType" TEXT,
    "isExecutable" BOOLEAN NOT NULL DEFAULT false,
    "content" BYTEA NOT NULL,

    CONSTRAINT "SkillListingFile_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "SkillListingFile_skillListingVersionId_relativePath_key" ON "SkillListingFile"("skillListingVersionId", "relativePath");

-- AddForeignKey
ALTER TABLE "SkillListingFile" ADD CONSTRAINT "SkillListingFile_skillListingVersionId_fkey" FOREIGN KEY ("skillListingVersionId") REFERENCES "SkillListingVersion"("id") ON DELETE CASCADE ON UPDATE CASCADE;
