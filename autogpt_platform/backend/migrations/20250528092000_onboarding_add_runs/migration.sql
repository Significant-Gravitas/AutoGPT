-- AlterEnum
ALTER TYPE "OnboardingStep" ADD VALUE 'RUN_AGENTS';

-- AlterTable
ALTER TABLE "UserOnboarding" ADD COLUMN "agentRuns" INTEGER NOT NULL DEFAULT 0;
