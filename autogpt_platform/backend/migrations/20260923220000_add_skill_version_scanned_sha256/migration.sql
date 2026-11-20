-- AlterTable
ALTER TABLE "SkillListingVersion" ADD COLUMN     "scannedSha256" TEXT[] DEFAULT ARRAY[]::TEXT[];
