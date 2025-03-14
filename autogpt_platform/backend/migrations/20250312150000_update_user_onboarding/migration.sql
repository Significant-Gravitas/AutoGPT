-- Create OnboardingStep enum
CREATE TYPE "OnboardingStep" AS ENUM (
  'WELCOME',
  'USAGE_REASON',
  'INTEGRATIONS',
  'AGENT_CHOICE',
  'AGENT_NEW_RUN',
  'AGENT_INPUT',
  'CONGRATS'
);

-- Modify the UserOnboarding table
ALTER TABLE "UserOnboarding"
  DROP COLUMN "step",
  DROP COLUMN "isCompleted",
  ADD COLUMN "completedSteps" "OnboardingStep"[] DEFAULT '{}';