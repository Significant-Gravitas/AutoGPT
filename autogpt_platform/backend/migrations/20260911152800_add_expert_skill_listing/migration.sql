-- CreateTable
CREATE TABLE "ExpertSkillListing" (
    "expertId" TEXT NOT NULL,
    "skillListingId" TEXT NOT NULL,
    "position" INTEGER NOT NULL DEFAULT 0,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "ExpertSkillListing_pkey" PRIMARY KEY ("expertId","skillListingId")
);

-- CreateIndex
CREATE INDEX "ExpertSkillListing_skillListingId_idx" ON "ExpertSkillListing"("skillListingId");

-- AddForeignKey
ALTER TABLE "ExpertSkillListing" ADD CONSTRAINT "ExpertSkillListing_expertId_fkey" FOREIGN KEY ("expertId") REFERENCES "Expert"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ExpertSkillListing" ADD CONSTRAINT "ExpertSkillListing_skillListingId_fkey" FOREIGN KEY ("skillListingId") REFERENCES "SkillListing"("id") ON DELETE CASCADE ON UPDATE CASCADE;

