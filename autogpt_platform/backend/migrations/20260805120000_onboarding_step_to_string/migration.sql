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

-- Postgres refuses to retype a column that a view depends on, and the
-- analytics views (analytics.user_onboarding, analytics.user_onboarding_funnel)
-- read these columns. They are applied out-of-band from analytics/queries/*.sql,
-- so they exist in deployed databases but not in one built from migrations
-- alone -- which is why this only fails on a real environment.
--
-- Capture whatever dependent views actually exist (definition + grants), drop
-- them, retype, then recreate them verbatim. Reading the definition from the
-- catalog rather than the repo means we restore exactly what was deployed, and
-- picks up any view not tracked in this repo.
CREATE TEMP TABLE _onboarding_dep_views AS
SELECT n.nspname AS schema_name,
       c.relname AS view_name,
       pg_get_userbyid(c.relowner) AS view_owner,
       rtrim(pg_get_viewdef(c.oid, true), E' \n;') AS definition
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind = 'v'
  AND EXISTS (
        SELECT 1
        FROM pg_depend d
        JOIN pg_rewrite r ON r.oid = d.objid AND r.ev_class = c.oid
        WHERE d.refobjid = 'platform."UserOnboarding"'::regclass
          AND d.refobjsubid > 0
      );

-- Read the ACL from the catalog rather than information_schema, whose
-- role_table_grants view only shows grants involving roles the current user
-- belongs to and would silently drop the rest.
CREATE TEMP TABLE _onboarding_dep_view_grants AS
SELECT v.schema_name,
       v.view_name,
       CASE WHEN a.grantee = 0 THEN 'PUBLIC'
            ELSE quote_ident(pg_get_userbyid(a.grantee)) END AS grantee_name,
       a.privilege_type,
       a.is_grantable
FROM _onboarding_dep_views v
JOIN pg_class c ON c.relname = v.view_name
JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = v.schema_name
CROSS JOIN LATERAL aclexplode(c.relacl) AS a;

DO $$
DECLARE v record;
BEGIN
    FOR v IN SELECT * FROM _onboarding_dep_views LOOP
        EXECUTE format('DROP VIEW %I.%I', v.schema_name, v.view_name);
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

-- Recreate the dependent views verbatim, restoring owner and grants. Their
-- step columns come back as TEXT[] instead of "OnboardingStep"[]; the queries
-- already cast with ::text, so consumers are unaffected.
DO $$
DECLARE v record;
        g record;
BEGIN
    FOR v IN SELECT * FROM _onboarding_dep_views LOOP
        EXECUTE format('CREATE VIEW %I.%I AS %s',
                       v.schema_name, v.view_name, v.definition);
        -- Ownership is cosmetic for readers; if the migration role isn't a
        -- member of the original owner it can't reassign. Warn rather than
        -- abort, so a privilege quirk can't strand the migration half-applied.
        BEGIN
            EXECUTE format('ALTER VIEW %I.%I OWNER TO %I',
                           v.schema_name, v.view_name, v.view_owner);
        EXCEPTION WHEN OTHERS THEN
            RAISE WARNING 'could not restore owner % on %.%: %',
                          v.view_owner, v.schema_name, v.view_name, SQLERRM;
        END;
    END LOOP;

    FOR g IN SELECT * FROM _onboarding_dep_view_grants LOOP
        BEGIN
            EXECUTE format('GRANT %s ON %I.%I TO %s%s',
                           g.privilege_type, g.schema_name, g.view_name,
                           g.grantee_name,
                           CASE WHEN g.is_grantable THEN ' WITH GRANT OPTION' ELSE '' END);
        EXCEPTION WHEN OTHERS THEN
            RAISE WARNING 'could not restore % on %.% to %: %',
                          g.privilege_type, g.schema_name, g.view_name,
                          g.grantee_name, SQLERRM;
        END;
    END LOOP;
END $$;

DROP TABLE _onboarding_dep_views;
DROP TABLE _onboarding_dep_view_grants;

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
