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
  DROP COLUMN "selectedAgentCreator",
  DROP COLUMN "selectedAgentSlug",
  ADD COLUMN "completedSteps" "OnboardingStep"[] DEFAULT '{}',
  ADD COLUMN "selectedStoreListingVersionId" TEXT
