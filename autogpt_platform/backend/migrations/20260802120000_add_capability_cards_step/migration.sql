-- Copilot home first-run: records that the capability-cards modal was
-- completed or skipped, so it shows only once per user.
ALTER TYPE "OnboardingStep" ADD VALUE 'CAPABILITY_CARDS';
