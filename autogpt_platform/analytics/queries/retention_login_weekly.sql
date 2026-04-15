-- =============================================================
-- View: analytics.retention_login_weekly
-- Looker source alias: ds83  |  Charts: 2
-- =============================================================
-- DESCRIPTION
--   Weekly cohort retention based on login sessions.
--   Users are grouped by the ISO week of their first ever login.
--   For each cohort × lifetime-week combination, outputs both:
--     - bounded rate: % active in exactly that week
--     - unbounded rate: % who were ever active on or after that week
--   Weeks are capped to the cohort's actual age (no future data points).
--
-- SOURCE TABLES
--   auth.sessions  — Login session records
--
-- HOW TO READ THE OUTPUT
--   cohort_week_start   The Monday of the week users first logged in
--   user_lifetime_week  0 = signup week, 1 = one week later, etc.
--   retention_rate_bounded   = active_users_bounded / cohort_users
--   retention_rate_unbounded = retained_users_unbounded / cohort_users
--
-- OUTPUT COLUMNS
--   cohort_week_start         DATE     First day of the cohort's signup week
--   cohort_label              TEXT     ISO week label (e.g. '2025-W01')
--   cohort_label_n            TEXT     ISO week label with cohort size (e.g. '2025-W01 (n=42)')
--   user_lifetime_week        INT      Weeks since first login (0 = signup week)
--   cohort_users              BIGINT   Total users in this cohort (denominator)
--   active_users_bounded      BIGINT   Users active in exactly week k
--   retained_users_unbounded  BIGINT   Users active any time on/after week k
--   retention_rate_bounded    FLOAT    bounded active / cohort_users
--   retention_rate_unbounded  FLOAT    unbounded retained / cohort_users
--   cohort_users_w0           BIGINT   cohort_users only at week 0, else 0 (safe to SUM in pivot tables)
--
-- EXAMPLE QUERIES
--   -- Week-1 retention rate per cohort
--   SELECT cohort_label, retention_rate_bounded AS w1_retention
--   FROM analytics.retention_login_weekly
--   WHERE user_lifetime_week = 1
--   ORDER BY cohort_week_start;
--
--   -- Overall average retention curve (all cohorts combined)
--   SELECT user_lifetime_week,
--          SUM(active_users_bounded)::float / NULLIF(SUM(cohort_users_w0), 0) AS avg_retention
--   FROM analytics.retention_login_weekly
--   GROUP BY 1 ORDER BY 1;
-- =============================================================

WITH params AS (SELECT 12::int AS max_weeks),
events AS (
  SELECT s.user_id::text AS user_id, s.created_at::timestamptz AS created_at,
         DATE_TRUNC('week', s.created_at)::date AS week_start
  FROM auth.sessions s WHERE s.user_id IS NOT NULL
),
first_login AS (
  SELECT user_id, MIN(created_at) AS first_login_time,
         DATE_TRUNC('week', MIN(created_at))::date AS cohort_week_start
  FROM events GROUP BY 1
),
activity_weeks AS (SELECT DISTINCT user_id, week_start FROM events),
user_week_age AS (
  SELECT aw.user_id, fl.cohort_week_start,
         ((aw.week_start - DATE_TRUNC('week', fl.first_login_time)::date) / 7)::int AS user_lifetime_week
  FROM activity_weeks aw JOIN first_login fl USING (user_id)
  WHERE aw.week_start >= DATE_TRUNC('week', fl.first_login_time)::date
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
cohort_sizes AS (SELECT cohort_week_start, COUNT(DISTINCT user_id) AS cohort_users FROM first_login GROUP BY 1),
cohort_caps AS (
  SELECT cs.cohort_week_start, cs.cohort_users,
         LEAST((SELECT max_weeks FROM params),
               GREATEST(0,((DATE_TRUNC('week',CURRENT_DATE)::date - cs.cohort_week_start)/7)::int)) AS cap_weeks
  FROM cohort_sizes cs
),
grid AS (
  SELECT cc.cohort_week_start, gs AS user_lifetime_week, cc.cohort_users
  FROM cohort_caps cc CROSS JOIN LATERAL generate_series(0, cc.cap_weeks) gs
)
SELECT
  g.cohort_week_start,
  TO_CHAR(g.cohort_week_start,'IYYY-"W"IW')                                    AS cohort_label,
  TO_CHAR(g.cohort_week_start,'IYYY-"W"IW')||' (n='||g.cohort_users||')'       AS cohort_label_n,
  g.user_lifetime_week, g.cohort_users,
  COALESCE(b.active_users_bounded,0)     AS active_users_bounded,
  COALESCE(u.retained_users_unbounded,0) AS retained_users_unbounded,
  CASE WHEN g.cohort_users>0 THEN COALESCE(b.active_users_bounded,0)::float/g.cohort_users END    AS retention_rate_bounded,
  CASE WHEN g.cohort_users>0 THEN COALESCE(u.retained_users_unbounded,0)::float/g.cohort_users END AS retention_rate_unbounded,
  CASE WHEN g.user_lifetime_week=0 THEN g.cohort_users ELSE 0 END               AS cohort_users_w0
FROM grid g
LEFT JOIN bounded_counts   b ON b.cohort_week_start=g.cohort_week_start AND b.user_lifetime_week=g.user_lifetime_week
LEFT JOIN unbounded_counts u ON u.cohort_week_start=g.cohort_week_start AND u.user_lifetime_week=g.user_lifetime_week
ORDER BY g.cohort_week_start, g.user_lifetime_week
