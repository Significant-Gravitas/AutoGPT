-- The copilot home's greeting reads these. Generated when the dump is
-- processed (during the onboarding loading screen) rather than on page
-- load, so landing on /copilot never waits on an LLM.
--
-- "greetingSeen" flips true the first time the user sends a message
-- (their first session). The greeting content is kept forever — only the
-- flag decides whether it is shown again.
ALTER TABLE "OnboardingBrainDump"
    ADD COLUMN "greeting" TEXT,
    ADD COLUMN "suggestedPrompts" JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN "greetingSeen" BOOLEAN NOT NULL DEFAULT false;
