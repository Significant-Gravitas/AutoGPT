/*
  Warnings:

  - You are about to drop the column `notificationDot` on the `UserOnboarding` table. All the data in the column will be lost.

*/
-- AlterEnum
-- This migration adds more than one value to an enum.
-- With PostgreSQL versions 11 and earlier, this is not possible
-- in a single migration. This can be worked around by creating
-- multiple migrations, each migration adding only one value to
-- the enum.


ALTER TYPE "OnboardingStep" ADD VALUE 'RE_RUN_AGENT';
ALTER TYPE "OnboardingStep" ADD VALUE 'SCHEDULE_AGENT';
ALTER TYPE "OnboardingStep" ADD VALUE 'RUN_3_DAYS';
ALTER TYPE "OnboardingStep" ADD VALUE 'TRIGGER_WEBHOOK';
ALTER TYPE "OnboardingStep" ADD VALUE 'RUN_14_DAYS';
ALTER TYPE "OnboardingStep" ADD VALUE 'RUN_AGENTS_100';

-- AlterTable
ALTER TABLE "UserOnboarding" DROP COLUMN "notificationDot",
ADD COLUMN     "consecutiveRunDays" INTEGER NOT NULL DEFAULT 0,
ADD COLUMN     "lastRunAt" TIMESTAMP(3),
ADD COLUMN     "walletShown" BOOLEAN NOT NULL DEFAULT false;
