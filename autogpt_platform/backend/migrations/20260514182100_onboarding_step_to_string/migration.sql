-- Convert UserOnboarding step columns from `OnboardingStep[]` to `String[]` so
-- that step renames/adds/retires become code-only changes. Boundary validation
-- moves to the API layer (Pydantic Literal on backend, OpenAPI-generated union
-- on frontend). See SECRT-2355.
--
-- Also renames the wizard-completion step `VISIT_COPILOT` -> `ONBOARDING_COMPLETE`
-- in existing rows so users who already completed onboarding keep their wallet
-- "Complete onboarding $3" tile and don't get re-routed through the wizard.

-- Drop defaults so the column type cast doesn't trip on the default's enum type.
ALTER TABLE "UserOnboarding" ALTER COLUMN "completedSteps" DROP DEFAULT;
ALTER TABLE "UserOnboarding" ALTER COLUMN "notified" DROP DEFAULT;
ALTER TABLE "UserOnboarding" ALTER COLUMN "rewardedFor" DROP DEFAULT;

-- Retype the columns. Data is preserved verbatim (enum -> text is lossless).
ALTER TABLE "UserOnboarding"
    ALTER COLUMN "completedSteps" TYPE TEXT[] USING "completedSteps"::TEXT[],
    ALTER COLUMN "notified"       TYPE TEXT[] USING "notified"::TEXT[],
    ALTER COLUMN "rewardedFor"    TYPE TEXT[] USING "rewardedFor"::TEXT[];

-- Restore defaults on the new column type.
ALTER TABLE "UserOnboarding" ALTER COLUMN "completedSteps" SET DEFAULT '{}';
ALTER TABLE "UserOnboarding" ALTER COLUMN "notified"       SET DEFAULT '{}';
ALTER TABLE "UserOnboarding" ALTER COLUMN "rewardedFor"    SET DEFAULT '{}';

-- Rename VISIT_COPILOT -> ONBOARDING_COMPLETE in existing rows. array_replace
-- is a no-op when the value isn't present, so this is safe on every row.
UPDATE "UserOnboarding"
SET    "completedSteps" = array_replace("completedSteps", 'VISIT_COPILOT', 'ONBOARDING_COMPLETE'),
       "notified"       = array_replace("notified",       'VISIT_COPILOT', 'ONBOARDING_COMPLETE'),
       "rewardedFor"    = array_replace("rewardedFor",    'VISIT_COPILOT', 'ONBOARDING_COMPLETE')
WHERE  'VISIT_COPILOT' = ANY("completedSteps")
   OR  'VISIT_COPILOT' = ANY("notified")
   OR  'VISIT_COPILOT' = ANY("rewardedFor");

-- Drop the now-unused enum type.
DROP TYPE "OnboardingStep";
