-- =============================================================
-- View: analytics.retention_execution_weekly
-- Looker source alias: ds92  |  Charts: 2
-- =============================================================
-- DESCRIPTION
--   Weekly cohort retention based on agent executions.
--   Cohort anchor = week of user's FIRST ever agent execution
--   (not first login). Only includes cohorts from the last 180 days.
--   Useful when you care about product engagement, not just visits.
--
-- SOURCE TABLES
--   platform.AgentGraphExecution  — Execution records
--
-- OUTPUT COLUMNS
--   Same pattern as retention_login_weekly.
--   cohort_week_start = week of first execution (not first login)
--
-- EXAMPLE QUERIES
--   -- Week-2 execution retention
--   SELECT cohort_label, retention_rate_bounded
--   FROM analytics.retention_execution_weekly
--   WHERE user_lifetime_week = 2 ORDER BY cohort_week_start;
-- =============================================================

WITH params AS (SELECT 12::int AS max_weeks, (CURRENT_DATE - INTERVAL '180 days') AS cohort_start),
events AS (
  SELECT e."userId"::text AS user_id, e."createdAt"::timestamptz AS created_at,
         DATE_TRUNC('week', e."createdAt")::date AS week_start
  FROM platform."AgentGraphExecution" e WHERE e."userId" IS NOT NULL
),
first_exec AS (
  SELECT user_id, MIN(created_at) AS first_exec_at,
         DATE_TRUNC('week', MIN(created_at))::date AS cohort_week_start
  FROM events GROUP BY 1
  HAVING MIN(created_at) >= (SELECT cohort_start FROM params)
),
activity_weeks AS (SELECT DISTINCT user_id, week_start FROM events),
user_week_age AS (
  SELECT aw.user_id, fe.cohort_week_start,
         ((aw.week_start - DATE_TRUNC('week',fe.first_exec_at)::date)/7)::int AS user_lifetime_week
  FROM activity_weeks aw JOIN first_exec fe USING (user_id)
  WHERE aw.week_start >= DATE_TRUNC('week',fe.first_exec_at)::date
),
bounded_counts AS (
  SELECT cohort_week_start, user_lifetime_week, COUNT(DISTINCT user_id) AS active_users_bounded
  FROM user_week_age WHERE user_lifetime_week >= 0 GROUP BY 1,2
),
last_active AS (
  SELECT cohort_week_start, user_id, MAX(user_lifetime_week) AS last_active_week FROM user_week_age GROUP BY 1,2
),
unbounded_counts AS (
  SELECT la.cohort_week_start, gs AS user_lifetime_week, COUNT(*) AS retained_users_unbounded
  FROM last_active la
  CROSS JOIN LATERAL generate_series(0, LEAST(la.last_active_week,(SELECT max_weeks FROM params))) gs
  GROUP BY 1,2
),
cohort_sizes AS (SELECT cohort_week_start, COUNT(DISTINCT user_id) AS cohort_users FROM first_exec GROUP BY 1),
cohort_caps AS (
  SELECT cs.cohort_week_start, cs.cohort_users,
         LEAST((SELECT max_weeks FROM params),
               GREATEST(0,((DATE_TRUNC('week',CURRENT_DATE)::date-cs.cohort_week_start)/7)::int)) AS cap_weeks
  FROM cohort_sizes cs
),
grid AS (
  SELECT cc.cohort_week_start, gs AS user_lifetime_week, cc.cohort_users
  FROM cohort_caps cc CROSS JOIN LATERAL generate_series(0, cc.cap_weeks) gs
)
SELECT
  g.cohort_week_start,
  TO_CHAR(g.cohort_week_start,'IYYY-"W"IW')                               AS cohort_label,
  TO_CHAR(g.cohort_week_start,'IYYY-"W"IW')||' (n='||g.cohort_users||')'  AS cohort_label_n,
  g.user_lifetime_week, g.cohort_users,
  COALESCE(b.active_users_bounded,0)     AS active_users_bounded,
  COALESCE(u.retained_users_unbounded,0) AS retained_users_unbounded,
  CASE WHEN g.cohort_users>0 THEN COALESCE(b.active_users_bounded,0)::float/g.cohort_users END    AS retention_rate_bounded,
  CASE WHEN g.cohort_users>0 THEN COALESCE(u.retained_users_unbounded,0)::float/g.cohort_users END AS retention_rate_unbounded,
  CASE WHEN g.user_lifetime_week=0 THEN g.cohort_users ELSE 0 END         AS cohort_users_w0
FROM grid g
LEFT JOIN bounded_counts   b ON b.cohort_week_start=g.cohort_week_start AND b.user_lifetime_week=g.user_lifetime_week
LEFT JOIN unbounded_counts u ON u.cohort_week_start=g.cohort_week_start AND u.user_lifetime_week=g.user_lifetime_week
ORDER BY g.cohort_week_start, g.user_lifetime_week;
