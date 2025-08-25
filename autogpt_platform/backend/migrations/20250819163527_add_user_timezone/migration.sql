-- AlterTable
ALTER TABLE "User" ADD COLUMN     "timezone" TEXT NOT NULL DEFAULT 'not-set' 
    CHECK (timezone = 'not-set' OR now() AT TIME ZONE timezone IS NOT NULL);
