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

-- Postgres refuses to retype a column that a view depends on, and the analytics
-- views (analytics.user_onboarding, analytics.user_onboarding_funnel) read
-- these columns. Those views are not part of the migration history: they are
-- applied out-of-band by `poetry run analytics-views`
-- (backend/scripts/generate_views.py) from analytics/queries/*.sql. They
-- therefore exist in deployed databases and never in one built from migrations
-- alone, which is why this only fails on a real environment.
--
-- Drop the dependents and let that script recreate them. Deliberately NOT
-- restoring them from the catalog here: the deployed definition of
-- user_onboarding_funnel still hardcodes the pre-rename step grid, so
-- recreating it verbatim would resurrect a funnel that reports 0 for the
-- renamed steps. Re-running the script rebuilds every view from the repo,
-- which already carries the new names, and re-grants analytics_readonly.
--
-- *** After deploying, run: poetry run analytics-views ***
--
-- Discovery is dynamic (and CASCADE'd) so views not tracked in this repo, and
-- any view layered on top of these, are handled too.

-- Fail fast rather than queue behind a long-running BI query holding a lock on
-- these views; the deploy can retry.
SET LOCAL lock_timeout = '30s';

DO $$
DECLARE v record;
BEGIN
    FOR v IN
        SELECT n.nspname AS schema_name, c.relname AS rel_name, c.relkind
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relkind IN ('v', 'm')
          AND EXISTS (
                SELECT 1
                FROM pg_depend d
                JOIN pg_rewrite r ON r.oid = d.objid
                WHERE d.classid = 'pg_rewrite'::regclass
                  AND r.ev_class = c.oid
                  AND d.refobjid = 'platform."UserOnboarding"'::regclass
                  AND d.refobjsubid > 0
              )
    LOOP
        RAISE NOTICE 'dropping dependent relation %.% (recreate with: poetry run analytics-views)',
                     v.schema_name, v.rel_name;
        IF v.relkind = 'm' THEN
            EXECUTE format('DROP MATERIALIZED VIEW IF EXISTS %I.%I CASCADE',
                           v.schema_name, v.rel_name);
        ELSE
            EXECUTE format('DROP VIEW IF EXISTS %I.%I CASCADE',
                           v.schema_name, v.rel_name);
        END IF;
    END LOOP;
END $$;

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
