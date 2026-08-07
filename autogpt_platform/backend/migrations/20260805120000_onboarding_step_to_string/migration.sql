-- Convert UserOnboarding step columns from `OnboardingStep[]` to `String[]` so
-- that step renames/adds/retires become code-only changes. Boundary validation
-- moves to the API layer (Pydantic Literal on backend, OpenAPI-generated union
-- on frontend). See SECRT-2355.
--
-- Also renames two steps in existing rows so users keep their progress/rewards:
--   * `VISIT_COPILOT`        -> `ONBOARDING_COMPLETE` (wizard-completion signal
--     backing the wallet's "Complete onboarding $3" tile; avoids re-routing
--     already-onboarded users through the wizard).
--   * `MARKETPLACE_RUN_AGENT` -> `LIBRARY_RUN_AGENT` (the step fires on Library
--     runs and is shown to users as a Library action, so the MARKETPLACE_ prefix
--     was a misnomer).

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

-- Rename retired step names in existing rows so users keep their progress:
-- VISIT_COPILOT -> ONBOARDING_COMPLETE and MARKETPLACE_RUN_AGENT ->
-- LIBRARY_RUN_AGENT. Both renames are chained into a single pass so the table
-- is rewritten once. array_replace is a no-op when the value isn't present, so
-- applying both to every matched row is safe.
UPDATE "UserOnboarding"
SET    "completedSteps" = array_replace(array_replace("completedSteps", 'VISIT_COPILOT', 'ONBOARDING_COMPLETE'), 'MARKETPLACE_RUN_AGENT', 'LIBRARY_RUN_AGENT'),
       "notified"       = array_replace(array_replace("notified",       'VISIT_COPILOT', 'ONBOARDING_COMPLETE'), 'MARKETPLACE_RUN_AGENT', 'LIBRARY_RUN_AGENT'),
       "rewardedFor"    = array_replace(array_replace("rewardedFor",    'VISIT_COPILOT', 'ONBOARDING_COMPLETE'), 'MARKETPLACE_RUN_AGENT', 'LIBRARY_RUN_AGENT')
WHERE  "completedSteps" && ARRAY['VISIT_COPILOT', 'MARKETPLACE_RUN_AGENT']
   OR  "notified"       && ARRAY['VISIT_COPILOT', 'MARKETPLACE_RUN_AGENT']
   OR  "rewardedFor"    && ARRAY['VISIT_COPILOT', 'MARKETPLACE_RUN_AGENT'];

-- Drop the now-unused enum type.
DROP TYPE "OnboardingStep";
