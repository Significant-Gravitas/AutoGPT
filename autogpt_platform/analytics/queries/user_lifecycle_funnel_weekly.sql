-- =============================================================
-- View: analytics.user_lifecycle_funnel_weekly
-- Looker source alias: (new)  |  Charts: 0
-- =============================================================
-- DESCRIPTION
--   Activation funnel per signup-week cohort, built on
--   analytics.user_lifecycle (so the definitions live in one place).
--   Each stage is a count of users in the cohort who reached it, plus
--   the share of the cohort. Stages that need time to mature (e.g.
--   activated within 14 days) are NULL until the whole cohort week has
--   aged that long, counted from the week's last day (so week start plus
--   7 days plus the window), so a half-baked recent week never reads as
--   a drop.
--
-- SOURCE VIEWS
--   analytics.user_lifecycle
--
-- OUTPUT COLUMNS
--   cohort_week_start       DATE    Monday of the signup week
--   cohort_label            TEXT    ISO week label
--   signups                 BIGINT  Users who signed up that week
--   onboarded               BIGINT  Completed onboarding
--   first_task_7d           BIGINT  Did a task within 7 days of signup
--   activated_14d           BIGINT  Met the activation definition (see user_lifecycle)
--   connected_integration   BIGINT  Connected at least one credential (any time)
--   created_schedule        BIGINT  Created at least one schedule (any time)
--   used_expert             BIGINT  Talked to or ran an expert (any time)
--   purchased               BIGINT  Bought credits or a subscription (any time)
--   retained_w4             BIGINT  Did a task in their fourth week after signup
--                                   (days 21-27; cohort >= 4 weeks old)
--   pct_onboarded, pct_first_task_7d, pct_activated_14d, pct_connected_integration,
--   pct_created_schedule, pct_used_expert, pct_purchased, pct_retained_w4
--                                                       FLOAT  share of signups
--
-- WINDOW
--   Signup cohorts from the last 180 days
--
-- EXAMPLE QUERIES
--   SELECT cohort_label, signups, pct_first_task_7d, pct_activated_14d, pct_retained_w4
--   FROM analytics.user_lifecycle_funnel_weekly ORDER BY cohort_week_start;
-- =============================================================

WITH cohorts AS (
  SELECT
    DATE_TRUNC('week', signup_at)::date                                  AS cohort_week_start,
    COUNT(*)                                                             AS signups,
    COUNT(*) FILTER (WHERE onboarding_completed)                         AS onboarded,
    COUNT(*) FILTER (WHERE first_task_within_7d)                         AS first_task_7d,
    COUNT(*) FILTER (WHERE activated)                                    AS activated_14d,
    COUNT(*) FILTER (WHERE integrations_connected_total > 0)             AS connected_integration,
    COUNT(*) FILTER (WHERE schedules_created_total > 0)                  AS created_schedule,
    COUNT(*) FILTER (WHERE expert_turns_total > 0 OR expert_workflow_runs_total > 0)
                                                                         AS used_expert,
    COUNT(*) FILTER (WHERE purchases_total > 0)                          AS purchased,
    COUNT(*) FILTER (WHERE tasks_week_4 > 0)                             AS retained_w4,
    MIN(signup_at)                                                       AS cohort_started_at
  FROM analytics.user_lifecycle
  WHERE signup_at >= CURRENT_DATE - INTERVAL '180 days'
  GROUP BY 1
)
SELECT
  cohort_week_start,
  TO_CHAR(cohort_week_start, 'IYYY-"W"IW')                               AS cohort_label,
  signups,
  onboarded,
  CASE WHEN cohort_week_start + 14 <= CURRENT_DATE THEN first_task_7d END AS first_task_7d,
  CASE WHEN cohort_week_start + 21 <= CURRENT_DATE THEN activated_14d END AS activated_14d,
  connected_integration,
  created_schedule,
  used_expert,
  purchased,
  CASE WHEN cohort_week_start + 35 <= CURRENT_DATE THEN retained_w4 END  AS retained_w4,
  onboarded::float / NULLIF(signups, 0)                                  AS pct_onboarded,
  CASE WHEN cohort_week_start + 14 <= CURRENT_DATE
       THEN first_task_7d::float / NULLIF(signups, 0) END                AS pct_first_task_7d,
  CASE WHEN cohort_week_start + 21 <= CURRENT_DATE
       THEN activated_14d::float / NULLIF(signups, 0) END                AS pct_activated_14d,
  connected_integration::float / NULLIF(signups, 0)                      AS pct_connected_integration,
  created_schedule::float / NULLIF(signups, 0)                           AS pct_created_schedule,
  used_expert::float / NULLIF(signups, 0)                                AS pct_used_expert,
  purchased::float / NULLIF(signups, 0)                                  AS pct_purchased,
  CASE WHEN cohort_week_start + 35 <= CURRENT_DATE
       THEN retained_w4::float / NULLIF(signups, 0) END                  AS pct_retained_w4
FROM cohorts
ORDER BY cohort_week_start
