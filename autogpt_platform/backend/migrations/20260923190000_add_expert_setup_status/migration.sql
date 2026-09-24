-- CreateEnum
CREATE TYPE "ExpertSetupStatus" AS ENUM ('INSTALLING', 'READY', 'FAILED');

-- AlterTable
ALTER TABLE "Expert" ADD COLUMN     "setupFailures" TEXT[] DEFAULT ARRAY[]::TEXT[],
ADD COLUMN     "setupStartedAt" TIMESTAMP(3),
ADD COLUMN     "setupStatus" "ExpertSetupStatus" NOT NULL DEFAULT 'READY';
