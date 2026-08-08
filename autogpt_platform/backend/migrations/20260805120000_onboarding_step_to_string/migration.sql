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
-- views read these columns. Those views are not part of the migration history:
-- they are applied out-of-band by `poetry run analytics-views`
-- (backend/scripts/generate_views.py) from analytics/queries/*.sql, so they
-- exist in deployed databases and never in one built from migrations alone --
-- which is why this only fails on a real environment.
--
-- Drop the dependents, retype, then recreate them here so no manual step is
-- needed after deploy. The definitions below are copied from
-- analytics/queries/*.sql as of this migration, matching what analytics-views
-- would produce (`WITH (security_invoker = false)`), and deliberately NOT
-- restored from the catalog: the deployed user_onboarding_funnel still
-- hardcodes the pre-rename step grid, so restoring it verbatim would resurrect
-- a funnel reporting 0 for ONBOARDING_COMPLETE and LIBRARY_RUN_AGENT.
--
-- Only views that were present are recreated, so a database without them
-- (local, CI) is unaffected. Anything dropped that is not recreated here --
-- a materialized view, or a dependent not tracked in this repo -- is reported
-- via RAISE WARNING so the operator knows to rebuild it.

-- Fail fast rather than queue behind a long-running BI query holding a lock on
-- these views; the deploy can retry.
SET LOCAL lock_timeout = '30s';

CREATE TEMP TABLE _onboarding_dropped_relations AS
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
          AND d.refobjid = '"UserOnboarding"'::regclass
          -- Only the columns actually being retyped. Postgres blocks the ALTER
          -- solely for views depending on those; a view reading other columns
          -- (e.g. analytics.user_onboarding_integration, which uses
          -- "integrations") is no obstacle and must not be dropped.
          AND d.refobjsubid IN (
                SELECT attnum
                FROM pg_attribute
                WHERE attrelid = '"UserOnboarding"'::regclass
                  AND attname IN ('completedSteps', 'notified', 'rewardedFor')
              )
      );

DO $$
DECLARE v record;
BEGIN
    FOR v IN SELECT * FROM _onboarding_dropped_relations LOOP
        RAISE NOTICE 'dropping dependent %: %.%',
                     CASE v.relkind WHEN 'm' THEN 'materialized view' ELSE 'view' END,
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

-- Recreate the tracked analytics views that were dropped above.
DO $$
DECLARE v record;
BEGIN
    IF EXISTS (SELECT 1 FROM _onboarding_dropped_relations
                WHERE schema_name = 'analytics' AND rel_name = 'user_onboarding'
                  AND relkind = 'v') THEN
        EXECUTE $view$
CREATE OR REPLACE VIEW analytics.user_onboarding WITH (security_invoker = false) AS
SELECT
    id,
    "createdAt",
    "updatedAt",
    "usageReason",
    integrations,
    "userId",
    "completedSteps",
    "selectedStoreListingVersionId"
FROM platform."UserOnboarding"
$view$;
    END IF;

    IF EXISTS (SELECT 1 FROM _onboarding_dropped_relations
                WHERE schema_name = 'analytics' AND rel_name = 'user_onboarding_funnel'
                  AND relkind = 'v') THEN
        EXECUTE $view$
CREATE OR REPLACE VIEW analytics.user_onboarding_funnel WITH (security_invoker = false) AS
WITH all_steps AS (
  -- Complete ordered grid of all 22 steps so zero-completion steps
  -- are always present, keeping LAG comparisons correct.
  SELECT step_name, step_order
  FROM (VALUES
    ('WELCOME',               1),
    ('USAGE_REASON',          2),
    ('INTEGRATIONS',          3),
    ('AGENT_CHOICE',          4),
    ('AGENT_NEW_RUN',         5),
    ('AGENT_INPUT',           6),
    ('CONGRATS',              7),
    ('GET_RESULTS',           8),
    ('MARKETPLACE_VISIT',     9),
    ('MARKETPLACE_ADD_AGENT', 10),
    ('LIBRARY_RUN_AGENT',     11),
    ('BUILDER_OPEN',          12),
    ('BUILDER_SAVE_AGENT',    13),
    ('BUILDER_RUN_AGENT',     14),
    ('ONBOARDING_COMPLETE',   15),
    ('RE_RUN_AGENT',          16),
    ('SCHEDULE_AGENT',        17),
    ('RUN_AGENTS',            18),
    ('RUN_3_DAYS',            19),
    ('TRIGGER_WEBHOOK',       20),
    ('RUN_14_DAYS',           21),
    ('RUN_AGENTS_100',        22)
  ) AS t(step_name, step_order)
),
raw AS (
  SELECT
      u."userId",
      step_txt::text AS step
  FROM platform."UserOnboarding" u
  CROSS JOIN LATERAL UNNEST(u."completedSteps") AS step_txt
  WHERE u."createdAt" >= CURRENT_DATE - INTERVAL '90 days'
),
step_counts AS (
  SELECT step, COUNT(DISTINCT "userId") AS users_completed
  FROM raw GROUP BY step
),
funnel AS (
  SELECT
      a.step_name                          AS step,
      a.step_order,
      COALESCE(sc.users_completed, 0)      AS users_completed,
      ROUND(
        100.0 * COALESCE(sc.users_completed, 0)
        / NULLIF(
            LAG(COALESCE(sc.users_completed, 0)) OVER (ORDER BY a.step_order),
            0
          ),
        2
      )                                    AS pct_from_prev
  FROM all_steps a
  LEFT JOIN step_counts sc ON sc.step = a.step_name
)
SELECT * FROM funnel ORDER BY step_order
$view$;
    END IF;

    -- analytics-views grants SELECT on the whole schema to analytics_readonly;
    -- mirror that for the views recreated here.
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'analytics_readonly')
       AND EXISTS (SELECT 1 FROM _onboarding_dropped_relations WHERE schema_name = 'analytics') THEN
        EXECUTE 'GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO analytics_readonly';
    END IF;

    -- Anything dropped that this migration cannot rebuild.
    FOR v IN
        SELECT * FROM _onboarding_dropped_relations
        WHERE NOT (schema_name = 'analytics'
                   AND rel_name IN ('user_onboarding', 'user_onboarding_funnel')
                   AND relkind = 'v')
    LOOP
        RAISE WARNING 'dropped %.% (%) and cannot rebuild it here -- recreate it manually',
                      v.schema_name, v.rel_name,
                      CASE v.relkind WHEN 'm' THEN 'materialized view' ELSE 'untracked view' END;
    END LOOP;
END $$;

DROP TABLE _onboarding_dropped_relations;

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
