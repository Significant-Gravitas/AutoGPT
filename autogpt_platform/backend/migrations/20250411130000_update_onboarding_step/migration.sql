-- Modify the OnboardingStep enum
ALTER TYPE "OnboardingStep" ADD VALUE 'GET_RESULTS';
ALTER TYPE "OnboardingStep" ADD VALUE 'MARKETPLACE_VISIT';
ALTER TYPE "OnboardingStep" ADD VALUE 'MARKETPLACE_ADD_AGENT';
ALTER TYPE "OnboardingStep" ADD VALUE 'MARKETPLACE_RUN_AGENT';
ALTER TYPE "OnboardingStep" ADD VALUE 'BUILDER_OPEN';
ALTER TYPE "OnboardingStep" ADD VALUE 'BUILDER_SAVE_AGENT';
ALTER TYPE "OnboardingStep" ADD VALUE 'BUILDER_RUN_AGENT';

-- Modify the UserOnboarding table
ALTER TABLE "UserOnboarding"
  ADD COLUMN "updatedAt" TIMESTAMP(3),
  ADD COLUMN "notificationDot" BOOLEAN NOT NULL DEFAULT true,
  ADD COLUMN "notified" "OnboardingStep"[] DEFAULT '{}',
  ADD COLUMN "rewardedFor" "OnboardingStep"[] DEFAULT '{}',
  ADD COLUMN "onboardingAgentExecutionId" TEXT
