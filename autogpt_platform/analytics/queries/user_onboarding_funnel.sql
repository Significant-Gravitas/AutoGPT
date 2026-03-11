-- =============================================================
-- View: analytics.user_onboarding_funnel
-- Looker source alias: ds74  |  Charts: 1
-- =============================================================
-- DESCRIPTION
--   Pre-aggregated onboarding funnel showing how many users
--   completed each step and the drop-off percentage from the
--   previous step.  One row per onboarding step.
--
-- SOURCE TABLES
--   platform.UserOnboarding  — Onboarding records with completedSteps array
--
-- OUTPUT COLUMNS
--   step             TEXT     Onboarding step enum name (e.g. 'WELCOME', 'CONGRATS')
--   step_order       INT      Numeric position in the funnel (1=first, 14=last)
--   users_completed  BIGINT   Distinct users who completed this step
--   pct_from_prev    NUMERIC  % of users from the previous step who reached this one
--
-- STEP ORDER
--   1  WELCOME               8  GET_RESULTS
--   2  USAGE_REASON          9  MARKETPLACE_VISIT
--   3  INTEGRATIONS         10  MARKETPLACE_ADD_AGENT
--   4  AGENT_CHOICE         11  MARKETPLACE_RUN_AGENT
--   5  AGENT_NEW_RUN        12  BUILDER_OPEN
--   6  AGENT_INPUT          13  BUILDER_SAVE_AGENT
--   7  CONGRATS             14  BUILDER_RUN_AGENT
--
-- WINDOW
--   Users who started onboarding in the last 90 days
--
-- EXAMPLE QUERIES
--   -- Full funnel
--   SELECT * FROM analytics.user_onboarding_funnel ORDER BY step_order;
--
--   -- Biggest drop-off point
--   SELECT step, pct_from_prev FROM analytics.user_onboarding_funnel
--   ORDER BY pct_from_prev ASC LIMIT 3;
-- =============================================================

WITH raw AS (
  SELECT
      u."userId",
      u."createdAt",
      step_txt AS step,
      CASE step_txt
           WHEN 'WELCOME'               THEN  1
           WHEN 'USAGE_REASON'          THEN  2
           WHEN 'INTEGRATIONS'          THEN  3
           WHEN 'AGENT_CHOICE'          THEN  4
           WHEN 'AGENT_NEW_RUN'         THEN  5
           WHEN 'AGENT_INPUT'           THEN  6
           WHEN 'CONGRATS'              THEN  7
           WHEN 'GET_RESULTS'           THEN  8
           WHEN 'MARKETPLACE_VISIT'     THEN  9
           WHEN 'MARKETPLACE_ADD_AGENT' THEN 10
           WHEN 'MARKETPLACE_RUN_AGENT' THEN 11
           WHEN 'BUILDER_OPEN'          THEN 12
           WHEN 'BUILDER_SAVE_AGENT'    THEN 13
           WHEN 'BUILDER_RUN_AGENT'     THEN 14
           WHEN 'VISIT_COPILOT'          THEN 15
           WHEN 'RE_RUN_AGENT'           THEN 16
           WHEN 'SCHEDULE_AGENT'         THEN 17
           WHEN 'RUN_AGENTS'             THEN 18
           WHEN 'RUN_3_DAYS'             THEN 19
           WHEN 'TRIGGER_WEBHOOK'        THEN 20
           WHEN 'RUN_14_DAYS'            THEN 21
           WHEN 'RUN_AGENTS_100'         THEN 22
      END AS step_order
  FROM platform."UserOnboarding" u
  CROSS JOIN LATERAL UNNEST(u."completedSteps") AS step_txt
  WHERE u."createdAt" >= CURRENT_DATE - INTERVAL '90 days'
),
step_counts AS (
  SELECT step, step_order, COUNT(DISTINCT "userId") AS users_completed
  FROM raw GROUP BY step, step_order
),
funnel AS (
  SELECT
      step, step_order, users_completed,
      ROUND(100.0 * users_completed /
            NULLIF(LAG(users_completed) OVER (ORDER BY step_order), 0), 2) AS pct_from_prev
  FROM step_counts
)
SELECT * FROM funnel ORDER BY step_order
