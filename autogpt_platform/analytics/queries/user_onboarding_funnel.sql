-- =============================================================
-- View: analytics.user_onboarding_funnel
-- Looker source alias: ds74  |  Charts: 1
-- =============================================================
-- DESCRIPTION
--   Pre-aggregated onboarding funnel showing how many users
--   completed each step and the drop-off percentage from the
--   previous step.  One row per onboarding step (all 22 steps
--   always present, even with 0 completions — prevents sparse
--   gaps from making LAG compare the wrong predecessors).
--
-- SOURCE TABLES
--   platform.UserOnboarding  — Onboarding records with completedSteps array
--
-- OUTPUT COLUMNS
--   step             TEXT     Onboarding step enum name (e.g. 'WELCOME', 'CONGRATS')
--   step_order       INT      Numeric position in the funnel (1=first, 22=last)
--   users_completed  BIGINT   Distinct users who completed this step
--   pct_from_prev    NUMERIC  % of users from the previous step who reached this one
--
-- STEP ORDER
--   1  WELCOME               9  MARKETPLACE_VISIT     17  SCHEDULE_AGENT
--   2  USAGE_REASON         10  MARKETPLACE_ADD_AGENT  18  RUN_AGENTS
--   3  INTEGRATIONS         11  MARKETPLACE_RUN_AGENT  19  RUN_3_DAYS
--   4  AGENT_CHOICE         12  BUILDER_OPEN           20  TRIGGER_WEBHOOK
--   5  AGENT_NEW_RUN        13  BUILDER_SAVE_AGENT     21  RUN_14_DAYS
--   6  AGENT_INPUT          14  BUILDER_RUN_AGENT      22  RUN_AGENTS_100
--   7  CONGRATS             15  VISIT_COPILOT
--   8  GET_RESULTS          16  RE_RUN_AGENT
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
    ('MARKETPLACE_RUN_AGENT', 11),
    ('BUILDER_OPEN',          12),
    ('BUILDER_SAVE_AGENT',    13),
    ('BUILDER_RUN_AGENT',     14),
    ('VISIT_COPILOT',         15),
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
