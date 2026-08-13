-- =============================================================
-- View: analytics.retention_agent
-- Looker source alias: ds35  |  Charts: 2
-- =============================================================
-- DESCRIPTION
--   Weekly cohort retention broken down per individual agent.
--   Cohort = week of a user's first use of THAT specific agent.
--   Tells you which agents keep users coming back vs. one-shot
--   use. Only includes cohorts from the last 180 days.
--
-- SOURCE TABLES
--   platform.AgentGraphExecution  — Execution records (user × agent × time)
--   platform.AgentGraph           — Agent names
--
-- OUTPUT COLUMNS
--   agent_id            TEXT   Agent graph UUID
--   agent_label         TEXT   'AgentName [first8chars]'
--   agent_label_n       TEXT   'AgentName [first8chars] (n=total_users)'
--   cohort_week_start   DATE   Week users first ran this agent
--   cohort_label        TEXT   ISO week label
--   cohort_label_n      TEXT   ISO week label with cohort size
--   user_lifetime_week  INT    Weeks since first use of this agent
--   cohort_users        BIGINT Users in this cohort for this agent
--   active_users        BIGINT Users who ran the agent again in week k
--   retention_rate      FLOAT  active_users / cohort_users
--   cohort_users_w0     BIGINT cohort_users only at week 0 (safe to SUM)
--   agent_total_users   BIGINT Total users across all cohorts for this agent
--
-- EXAMPLE QUERIES
--   -- Best-retained agents at week 2
--   SELECT agent_label, AVG(retention_rate) AS w2_retention
--   FROM analytics.retention_agent
--   WHERE user_lifetime_week = 2 AND cohort_users >= 10
--   GROUP BY 1 ORDER BY w2_retention DESC LIMIT 10;
--
--   -- Agents with most unique users
--   SELECT DISTINCT agent_label, agent_total_users
--   FROM analytics.retention_agent
--   ORDER BY agent_total_users DESC LIMIT 20;
-- =============================================================

WITH params AS (SELECT 12::int AS max_weeks, (CURRENT_DATE - INTERVAL '180 days') AS cohort_start),
events AS (
  SELECT e."userId"::text AS user_id, e."agentGraphId" AS agent_id,
         e."createdAt"::timestamptz AS created_at,
         DATE_TRUNC('week', e."createdAt")::date AS week_start
  FROM platform."AgentGraphExecution" e
),
first_use AS (
  SELECT user_id, agent_id, MIN(created_at) AS first_use_at,
         DATE_TRUNC('week', MIN(created_at))::date AS cohort_week_start
  FROM events GROUP BY 1,2
  HAVING MIN(created_at) >= (SELECT cohort_start FROM params)
),
activity_weeks AS (SELECT DISTINCT user_id, agent_id, week_start FROM events),
user_week_age AS (
  SELECT aw.user_id, aw.agent_id, fu.cohort_week_start,
         ((aw.week_start - DATE_TRUNC('week',fu.first_use_at)::date)/7)::int AS user_lifetime_week
  FROM activity_weeks aw JOIN first_use fu USING (user_id, agent_id)
  WHERE aw.week_start >= DATE_TRUNC('week',fu.first_use_at)::date
),
active_counts AS (
  SELECT agent_id, cohort_week_start, user_lifetime_week, COUNT(DISTINCT user_id) AS active_users
  FROM user_week_age WHERE user_lifetime_week >= 0 GROUP BY 1,2,3
),
cohort_sizes AS (
  SELECT agent_id, cohort_week_start, COUNT(DISTINCT user_id) AS cohort_users FROM first_use GROUP BY 1,2
),
cohort_caps AS (
  SELECT cs.agent_id, cs.cohort_week_start, cs.cohort_users,
         LEAST((SELECT max_weeks FROM params),
               GREATEST(0,((DATE_TRUNC('week',CURRENT_DATE)::date-cs.cohort_week_start)/7)::int)) AS cap_weeks
  FROM cohort_sizes cs
),
grid AS (
  SELECT cc.agent_id, cc.cohort_week_start, gs AS user_lifetime_week, cc.cohort_users
  FROM cohort_caps cc CROSS JOIN LATERAL generate_series(0, cc.cap_weeks) gs
),
agent_names AS (SELECT DISTINCT ON (g."id") g."id" AS agent_id, g."name" AS agent_name FROM platform."AgentGraph" g ORDER BY g."id", g."version" DESC),
agent_total_users AS (SELECT agent_id, SUM(cohort_users) AS agent_total_users FROM cohort_sizes GROUP BY 1)
SELECT
  g.agent_id,
  COALESCE(an.agent_name,'(unnamed)')||' ['||LEFT(g.agent_id::text,8)||']'  AS agent_label,
  COALESCE(an.agent_name,'(unnamed)')||' ['||LEFT(g.agent_id::text,8)||'] (n='||COALESCE(atu.agent_total_users,0)||')' AS agent_label_n,
  g.cohort_week_start,
  TO_CHAR(g.cohort_week_start,'IYYY-"W"IW')                               AS cohort_label,
  TO_CHAR(g.cohort_week_start,'IYYY-"W"IW')||' (n='||g.cohort_users||')'  AS cohort_label_n,
  g.user_lifetime_week, g.cohort_users,
  COALESCE(ac.active_users,0)                                              AS active_users,
  COALESCE(ac.active_users,0)::float / NULLIF(g.cohort_users,0)           AS retention_rate,
  CASE WHEN g.user_lifetime_week=0 THEN g.cohort_users ELSE 0 END         AS cohort_users_w0,
  COALESCE(atu.agent_total_users,0)                                        AS agent_total_users
FROM grid g
LEFT JOIN active_counts     ac  ON ac.agent_id=g.agent_id AND ac.cohort_week_start=g.cohort_week_start AND ac.user_lifetime_week=g.user_lifetime_week
LEFT JOIN agent_names       an  ON an.agent_id=g.agent_id
LEFT JOIN agent_total_users atu ON atu.agent_id=g.agent_id
ORDER BY agent_label, g.cohort_week_start, g.user_lifetime_week;
