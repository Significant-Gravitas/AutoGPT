-- =============================================================
-- View: analytics.retention_login_daily
-- Looker source alias: ds112  |  Charts: 1
-- =============================================================
-- DESCRIPTION
--   Daily cohort retention based on login sessions.
--   Same logic as retention_login_weekly but at day granularity,
--   showing up to day 30 for cohorts from the last 90 days.
--   Useful for analysing early activation (days 1-7) in detail.
--
-- SOURCE TABLES
--   auth.sessions  — Login session records
--
-- OUTPUT COLUMNS (same pattern as retention_login_weekly)
--   cohort_day_start          DATE     First day the cohort logged in
--   cohort_label              TEXT     Date string (e.g. '2025-03-01')
--   cohort_label_n            TEXT     Date + cohort size (e.g. '2025-03-01 (n=12)')
--   user_lifetime_day         INT      Days since first login (0 = signup day)
--   cohort_users              BIGINT   Total users in cohort
--   active_users_bounded      BIGINT   Users active on exactly day k
--   retained_users_unbounded  BIGINT   Users active any time on/after day k
--   retention_rate_bounded    FLOAT    bounded / cohort_users
--   retention_rate_unbounded  FLOAT    unbounded / cohort_users
--   cohort_users_d0           BIGINT   cohort_users only at day 0, else 0 (safe to SUM)
--
-- EXAMPLE QUERIES
--   -- Day-1 retention rate (came back next day)
--   SELECT cohort_label, retention_rate_bounded AS d1_retention
--   FROM analytics.retention_login_daily
--   WHERE user_lifetime_day = 1 ORDER BY cohort_day_start;
--
--   -- Average retention curve across all cohorts
--   SELECT user_lifetime_day,
--          SUM(active_users_bounded)::float / NULLIF(SUM(cohort_users_d0), 0) AS avg_retention
--   FROM analytics.retention_login_daily
--   GROUP BY 1 ORDER BY 1;
-- =============================================================

WITH params AS (SELECT 30::int AS max_days, (CURRENT_DATE - INTERVAL '90 days')::date AS cohort_start),
events AS (
  SELECT s.user_id::text AS user_id, s.created_at::timestamptz AS created_at,
         DATE_TRUNC('day', s.created_at)::date AS day_start
  FROM auth.sessions s WHERE s.user_id IS NOT NULL
),
first_login AS (
  SELECT user_id, MIN(created_at) AS first_login_time,
         DATE_TRUNC('day', MIN(created_at))::date AS cohort_day_start
  FROM events GROUP BY 1
  HAVING MIN(created_at) >= (SELECT cohort_start FROM params)
),
activity_days AS (SELECT DISTINCT user_id, day_start FROM events),
user_day_age AS (
  SELECT ad.user_id, fl.cohort_day_start,
         (ad.day_start - DATE_TRUNC('day', fl.first_login_time)::date)::int AS user_lifetime_day
  FROM activity_days ad JOIN first_login fl USING (user_id)
  WHERE ad.day_start >= DATE_TRUNC('day', fl.first_login_time)::date
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
cohort_sizes AS (SELECT cohort_day_start, COUNT(DISTINCT user_id) AS cohort_users FROM first_login GROUP BY 1),
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
  TO_CHAR(g.cohort_day_start,'YYYY-MM-DD')                                  AS cohort_label,
  TO_CHAR(g.cohort_day_start,'YYYY-MM-DD')||' (n='||g.cohort_users||')'     AS cohort_label_n,
  g.user_lifetime_day, g.cohort_users,
  COALESCE(b.active_users_bounded,0)     AS active_users_bounded,
  COALESCE(u.retained_users_unbounded,0) AS retained_users_unbounded,
  CASE WHEN g.cohort_users>0 THEN COALESCE(b.active_users_bounded,0)::float/g.cohort_users END    AS retention_rate_bounded,
  CASE WHEN g.cohort_users>0 THEN COALESCE(u.retained_users_unbounded,0)::float/g.cohort_users END AS retention_rate_unbounded,
  CASE WHEN g.user_lifetime_day=0 THEN g.cohort_users ELSE 0 END            AS cohort_users_d0
FROM grid g
LEFT JOIN bounded_counts   b ON b.cohort_day_start=g.cohort_day_start AND b.user_lifetime_day=g.user_lifetime_day
LEFT JOIN unbounded_counts u ON u.cohort_day_start=g.cohort_day_start AND u.user_lifetime_day=g.user_lifetime_day
ORDER BY g.cohort_day_start, g.user_lifetime_day;
