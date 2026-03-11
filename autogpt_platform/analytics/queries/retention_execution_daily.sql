-- =============================================================
-- View: analytics.retention_execution_daily
-- Looker source alias: ds111  |  Charts: 1
-- =============================================================
-- DESCRIPTION
--   Daily cohort retention based on agent executions.
--   Cohort anchor = day of user's FIRST ever execution.
--   Only includes cohorts from the last 90 days, up to day 30.
--   Great for early engagement analysis (did users run another
--   agent the next day?).
--
-- SOURCE TABLES
--   platform.AgentGraphExecution  — Execution records
--
-- OUTPUT COLUMNS
--   Same pattern as retention_login_daily.
--   cohort_day_start = day of first execution (not first login)
--
-- EXAMPLE QUERIES
--   -- Day-3 execution retention
--   SELECT cohort_label, retention_rate_bounded AS d3_retention
--   FROM analytics.retention_execution_daily
--   WHERE user_lifetime_day = 3 ORDER BY cohort_day_start;
-- =============================================================

WITH params AS (SELECT 30::int AS max_days, (CURRENT_DATE - INTERVAL '90 days') AS cohort_start),
events AS (
  SELECT e."userId"::text AS user_id, e."createdAt"::timestamptz AS created_at,
         DATE_TRUNC('day', e."createdAt")::date AS day_start
  FROM platform."AgentGraphExecution" e WHERE e."userId" IS NOT NULL
),
first_exec AS (
  SELECT user_id, MIN(created_at) AS first_exec_at,
         DATE_TRUNC('day', MIN(created_at))::date AS cohort_day_start
  FROM events GROUP BY 1
  HAVING MIN(created_at) >= (SELECT cohort_start FROM params)
),
activity_days AS (SELECT DISTINCT user_id, day_start FROM events),
user_day_age AS (
  SELECT ad.user_id, fe.cohort_day_start,
         (ad.day_start - DATE_TRUNC('day',fe.first_exec_at)::date)::int AS user_lifetime_day
  FROM activity_days ad JOIN first_exec fe USING (user_id)
  WHERE ad.day_start >= DATE_TRUNC('day',fe.first_exec_at)::date
),
bounded_counts AS (
  SELECT cohort_day_start, user_lifetime_day, COUNT(DISTINCT user_id) AS active_users_bounded
  FROM user_day_age WHERE user_lifetime_day >= 0 GROUP BY 1,2
),
last_active AS (
  SELECT cohort_day_start, user_id, MAX(user_lifetime_day) AS last_active_day FROM user_day_age GROUP BY 1,2
),
unbounded_counts AS (
  SELECT la.cohort_day_start, gs AS user_lifetime_day, COUNT(*) AS retained_users_unbounded
  FROM last_active la
  CROSS JOIN LATERAL generate_series(0, LEAST(la.last_active_day,(SELECT max_days FROM params))) gs
  GROUP BY 1,2
),
cohort_sizes AS (SELECT cohort_day_start, COUNT(DISTINCT user_id) AS cohort_users FROM first_exec GROUP BY 1),
cohort_caps AS (
  SELECT cs.cohort_day_start, cs.cohort_users,
         LEAST((SELECT max_days FROM params), GREATEST(0,(CURRENT_DATE-cs.cohort_day_start)::int)) AS cap_days
  FROM cohort_sizes cs
),
grid AS (
  SELECT cc.cohort_day_start, gs AS user_lifetime_day, cc.cohort_users
  FROM cohort_caps cc CROSS JOIN LATERAL generate_series(0, cc.cap_days) gs
)
SELECT
  g.cohort_day_start,
  TO_CHAR(g.cohort_day_start,'YYYY-MM-DD')                                AS cohort_label,
  TO_CHAR(g.cohort_day_start,'YYYY-MM-DD')||' (n='||g.cohort_users||')'   AS cohort_label_n,
  g.user_lifetime_day, g.cohort_users,
  COALESCE(b.active_users_bounded,0)     AS active_users_bounded,
  COALESCE(u.retained_users_unbounded,0) AS retained_users_unbounded,
  CASE WHEN g.cohort_users>0 THEN COALESCE(b.active_users_bounded,0)::float/g.cohort_users END    AS retention_rate_bounded,
  CASE WHEN g.cohort_users>0 THEN COALESCE(u.retained_users_unbounded,0)::float/g.cohort_users END AS retention_rate_unbounded,
  CASE WHEN g.user_lifetime_day=0 THEN g.cohort_users ELSE 0 END          AS cohort_users_d0
FROM grid g
LEFT JOIN bounded_counts   b ON b.cohort_day_start=g.cohort_day_start AND b.user_lifetime_day=g.user_lifetime_day
LEFT JOIN unbounded_counts u ON u.cohort_day_start=g.cohort_day_start AND u.user_lifetime_day=g.user_lifetime_day
ORDER BY g.cohort_day_start, g.user_lifetime_day;
