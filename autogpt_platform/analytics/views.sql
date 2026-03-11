-- =============================================================
-- AutoGPT Analytics Views for PostHog Data Warehouse
-- Run in Supabase SQL Editor as superuser (postgres role)
--
-- Creates an `analytics` schema with one view per Looker data
-- source. PostHog syncs these views instead of raw tables.
-- =============================================================

-- ── Schema ────────────────────────────────────────────────────
CREATE SCHEMA IF NOT EXISTS analytics;

-- ── Auth schema grants (run as postgres superuser) ────────────
GRANT USAGE ON SCHEMA auth TO analytics_readonly;
GRANT SELECT ON auth.sessions TO analytics_readonly;
GRANT SELECT ON auth.audit_log_entries TO analytics_readonly;

-- ── Platform schema grants ────────────────────────────────────
GRANT USAGE ON SCHEMA platform TO analytics_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA platform TO analytics_readonly;

-- ── Analytics schema grants ───────────────────────────────────
GRANT USAGE ON SCHEMA analytics TO analytics_readonly;


-- =============================================================
-- 1. Auth Activities  (ds49 · 1 chart)
-- =============================================================
CREATE OR REPLACE VIEW analytics.auth_activities AS
SELECT
    created_at,
    payload->>'actor_id'      AS actor_id,
    payload->>'actor_via_sso' AS actor_via_sso,
    payload->>'action'        AS action
FROM auth.audit_log_entries
WHERE created_at >= NOW() - INTERVAL '90 days';


-- =============================================================
-- 2. Users Activities  (ds56 · 5 charts)
-- =============================================================
CREATE OR REPLACE VIEW analytics.users_activities AS
WITH user_logins AS (
  SELECT
    user_id::text                                    AS user_id,
    MIN(created_at)                                  AS first_login_time,
    MAX(created_at)                                  AS last_login_time,
    GREATEST(
      MAX(refreshed_at)::timestamptz,
      MAX(created_at)::timestamptz
    )                                                AS last_visit_time
  FROM auth.sessions
  GROUP BY user_id
),
user_agents AS (
  SELECT
    ul.user_id,
    MAX(g."createdAt")  AS last_agent_save_time,
    COUNT(g."id")       AS agent_count
  FROM user_logins ul
  LEFT JOIN platform."AgentGraph" g
         ON ul.user_id = g."userId" AND g."isActive"
  GROUP BY ul.user_id
),
user_graph_runs AS (
  SELECT
    ul.user_id,
    MIN(e."createdAt")                    AS first_agent_run_time,
    MAX(e."createdAt")                    AS last_agent_run_time,
    COUNT(DISTINCT e."agentGraphId")      AS unique_agent_runs,
    COUNT(e."id")                         AS agent_runs
  FROM user_logins ul
  LEFT JOIN platform."AgentGraphExecution" e ON ul.user_id = e."userId"
  GROUP BY ul.user_id
),
user_node_runs AS (
  SELECT
    ul.user_id,
    COUNT(*)                                                        AS node_execution_count,
    COUNT(*) FILTER (WHERE n."executionStatus" = 'FAILED')          AS node_execution_failed,
    COUNT(*) FILTER (WHERE n."executionStatus" = 'COMPLETED')       AS node_execution_completed,
    COUNT(*) FILTER (WHERE n."executionStatus" = 'TERMINATED')      AS node_execution_terminated,
    COUNT(*) FILTER (WHERE n."executionStatus" = 'QUEUED')          AS node_execution_queued,
    COUNT(*) FILTER (WHERE n."executionStatus" = 'RUNNING')         AS node_execution_running
  FROM user_logins ul
  LEFT JOIN platform."AgentGraphExecution" g  ON ul.user_id = g."userId"
  LEFT JOIN platform."AgentNodeExecution"  n  ON g."id" = n."agentGraphExecutionId"
  GROUP BY ul.user_id
)
SELECT
  ul.*,
  ua.last_agent_save_time,
  ua.agent_count,
  gr.first_agent_run_time,
  gr.last_agent_run_time,
  gr.unique_agent_runs,
  gr.agent_runs,
  nr.node_execution_count,
  nr.node_execution_failed,
  nr.node_execution_completed,
  nr.node_execution_terminated,
  nr.node_execution_queued,
  nr.node_execution_running,
  CASE
    WHEN ul.first_login_time < NOW() - INTERVAL '7 days'
     AND ul.last_visit_time  >= ul.first_login_time + INTERVAL '7 days' THEN 1
    WHEN ul.first_login_time < NOW() - INTERVAL '7 days'
     AND ul.last_visit_time  <  ul.first_login_time + INTERVAL '7 days' THEN 0
    ELSE NULL
  END AS is_active_after_7d
FROM user_logins ul
LEFT JOIN user_agents     ua ON ul.user_id = ua.user_id
LEFT JOIN user_graph_runs gr ON ul.user_id = gr.user_id
LEFT JOIN user_node_runs  nr ON ul.user_id = nr.user_id;


-- =============================================================
-- 3. Graph Execution  (ds16 · 21 charts)
-- =============================================================
CREATE OR REPLACE VIEW analytics.graph_execution AS
SELECT
    ge."id"                                                        AS id,
    ge."agentGraphId"                                              AS agentGraphId,
    ge."agentGraphVersion"                                         AS agentGraphVersion,
    CASE
        WHEN jsonb_exists(ge."stats"::jsonb, 'error')
         AND (
               (ge."stats"::jsonb->>'error') ILIKE '%insufficient balance%'
            OR (ge."stats"::jsonb->>'error') ILIKE '%you have no credits left%'
             )
        THEN 'NO_CREDITS'
        ELSE CAST(ge."executionStatus" AS TEXT)
    END                                                            AS executionStatus,
    ge."createdAt"                                                 AS createdAt,
    ge."updatedAt"                                                 AS updatedAt,
    ge."userId"                                                    AS userId,
    g."name"                                                       AS agentGraphName,
    (ge."stats"::jsonb->>'cputime')::decimal                       AS cputime,
    (ge."stats"::jsonb->>'walltime')::decimal                      AS walltime,
    (ge."stats"::jsonb->>'node_count')::decimal                    AS node_count,
    (ge."stats"::jsonb->>'nodes_cputime')::decimal                 AS nodes_cputime,
    (ge."stats"::jsonb->>'nodes_walltime')::decimal                AS nodes_walltime,
    (ge."stats"::jsonb->>'cost')::decimal                          AS execution_cost,
    (ge."stats"::jsonb->>'correctness_score')::float               AS correctness_score,
    COALESCE(la.possibly_ai, FALSE)                                AS possibly_ai,
    REGEXP_REPLACE(
        REGEXP_REPLACE(
            TRIM(BOTH '"' FROM ge."stats"::jsonb->>'error'),
            '(https?://)([A-Za-z0-9.-]+)(:[0-9]+)?(/[^\s]*)?',
            '\1\2/...', 'gi'
        ),
        '[a-zA-Z0-9_:-]*\d[a-zA-Z0-9_:-]*', '*', 'g'
    )                                                              AS groupedErrorMessage
FROM platform."AgentGraphExecution" ge
LEFT JOIN platform."AgentGraph" g
       ON ge."agentGraphId" = g."id"
      AND ge."agentGraphVersion" = g."version"
LEFT JOIN (
    SELECT DISTINCT ON ("userId", "agentGraphId")
           "userId", "agentGraphId",
           ("settings"::jsonb->>'sensitive_action_safe_mode')::boolean AS possibly_ai
    FROM platform."LibraryAgent"
    WHERE "isDeleted"  = FALSE
      AND "isArchived" = FALSE
      AND ("settings"::jsonb->>'sensitive_action_safe_mode')::boolean = TRUE
    ORDER BY "userId", "agentGraphId", "agentGraphVersion" DESC
) la ON la."userId" = ge."userId" AND la."agentGraphId" = ge."agentGraphId"
WHERE ge."createdAt" > CURRENT_DATE - INTERVAL '90 days';


-- =============================================================
-- 4. Node Block Execution  (ds14 · 11 charts)
-- =============================================================
CREATE OR REPLACE VIEW analytics.node_block_execution AS
SELECT
    ne."id"                                                            AS id,
    ne."agentGraphExecutionId"                                         AS agentGraphExecutionId,
    ne."agentNodeId"                                                   AS agentNodeId,
    CAST(ne."executionStatus" AS TEXT)                                 AS executionStatus,
    ne."addedTime"                                                     AS addedTime,
    ne."queuedTime"                                                    AS queuedTime,
    ne."startedTime"                                                   AS startedTime,
    ne."endedTime"                                                     AS endedTime,
    (ne."stats"::jsonb->>'input_size')::bigint                         AS inputSize,
    (ne."stats"::jsonb->>'output_size')::bigint                        AS outputSize,
    (ne."stats"::jsonb->>'walltime')::numeric                          AS walltime,
    (ne."stats"::jsonb->>'cputime')::numeric                           AS cputime,
    (ne."stats"::jsonb->>'llm_retry_count')::int                       AS llmRetryCount,
    (ne."stats"::jsonb->>'llm_call_count')::int                        AS llmCallCount,
    (ne."stats"::jsonb->>'input_token_count')::bigint                  AS inputTokenCount,
    (ne."stats"::jsonb->>'output_token_count')::bigint                 AS outputTokenCount,
    b."name"                                                           AS blockName,
    b."id"                                                             AS blockId,
    REGEXP_REPLACE(
        REGEXP_REPLACE(
            TRIM(BOTH '"' FROM eio."data"::text),
            '(https?://)([A-Za-z0-9.-]+)(:[0-9]+)?(/[^\s]*)?',
            '\1\2/...', 'gi'
        ),
        '[a-zA-Z0-9_:-]*\d[a-zA-Z0-9_:-]*', '*', 'g'
    )                                                                  AS groupedErrorMessage,
    eio."data"                                                         AS errorMessage
FROM platform."AgentNodeExecution" ne
LEFT JOIN platform."AgentNode" nd
       ON ne."agentNodeId" = nd."id"
LEFT JOIN platform."AgentBlock" b
       ON nd."agentBlockId" = b."id"
LEFT JOIN platform."AgentNodeExecutionInputOutput" eio
       ON eio."referencedByOutputExecId" = ne."id"
      AND eio."name" = 'error'
      AND ne."executionStatus" = 'FAILED'
WHERE ne."addedTime" > CURRENT_DATE - INTERVAL '90 days';


-- =============================================================
-- 5. User Block Spending  (ds6 · 5 charts)
-- =============================================================
CREATE OR REPLACE VIEW analytics.user_block_spending AS
SELECT
    c."transactionKey"                                        AS transactionKey,
    c."userId"                                                AS userId,
    c."amount"                                                AS amount,
    c."amount" * -1                                           AS negativeAmount,
    c."type"                                                  AS transactionType,
    c."createdAt"                                             AS transactionTime,
    c.metadata->>'block_id'                                   AS blockId,
    c.metadata->>'block'                                      AS blockName,
    c.metadata->'input'->'credentials'->>'provider'           AS llm_provider,
    c.metadata->'input'->>'model'                             AS llm_model,
    c.metadata->>'node_exec_id'                               AS node_exec_id,
    (ne."stats"->'llm_call_count')::int                       AS llm_call_count,
    (ne."stats"->'llm_retry_count')::int                      AS llm_retry_count,
    (ne."stats"->'input_token_count')::int                    AS llm_input_token_count,
    (ne."stats"->'output_token_count')::int                   AS llm_output_token_count
FROM platform."CreditTransaction" c
LEFT JOIN platform."AgentNodeExecution" ne
       ON (c.metadata->>'node_exec_id') = ne."id"::text
WHERE c."createdAt" > CURRENT_DATE - INTERVAL '90 days';


-- =============================================================
-- 6. User Onboarding  (ds68 · 3 charts)
-- =============================================================
CREATE OR REPLACE VIEW analytics.user_onboarding AS
SELECT
    id,
    "createdAt",
    "updatedAt",
    "usageReason",
    integrations,
    "userId",
    "completedSteps",
    "selectedStoreListingVersionId"
FROM platform."UserOnboarding";


-- =============================================================
-- 7. User Onboarding Funnel  (ds74 · 1 chart)
-- =============================================================
CREATE OR REPLACE VIEW analytics.user_onboarding_funnel AS
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
      END AS step_order
  FROM platform."UserOnboarding" u
  CROSS JOIN LATERAL UNNEST(u."completedSteps") AS step_txt
  WHERE u."createdAt" >= CURRENT_DATE - INTERVAL '90 days'
),
step_counts AS (
  SELECT step, step_order, COUNT(DISTINCT "userId") AS users_completed
  FROM raw
  GROUP BY step, step_order
),
funnel AS (
  SELECT
      step,
      step_order,
      users_completed,
      ROUND(100.0 * users_completed /
            NULLIF(LAG(users_completed) OVER (ORDER BY step_order), 0), 2) AS pct_from_prev
  FROM step_counts
)
SELECT * FROM funnel ORDER BY step_order;


-- =============================================================
-- 8. User Onboarding Integration  (ds75 · 1 chart)
-- =============================================================
CREATE OR REPLACE VIEW analytics.user_onboarding_integration AS
WITH exploded AS (
  SELECT
      u."userId" AS user_id,
      UNNEST(u."integrations") AS integration
  FROM platform."UserOnboarding" u
  WHERE u."createdAt" >= CURRENT_DATE - INTERVAL '90 days'
)
SELECT
    integration,
    COUNT(DISTINCT user_id) AS users_with_integration
FROM exploded
WHERE integration IS NOT NULL AND integration <> ''
GROUP BY integration
ORDER BY users_with_integration DESC;


-- =============================================================
-- 9. Users Retention - Login Event (Weekly)  (ds83 · 2 charts)
-- =============================================================
CREATE OR REPLACE VIEW analytics.retention_login_weekly AS
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
ORDER BY g.cohort_week_start, g.user_lifetime_week;


-- =============================================================
-- 10. Users Retention - Login Event (Daily)  (ds112 · 1 chart)
-- =============================================================
CREATE OR REPLACE VIEW analytics.retention_login_daily AS
WITH params AS (SELECT 30::int AS max_days),
events AS (
  SELECT s.user_id::text AS user_id, s.created_at::timestamptz AS created_at,
         DATE_TRUNC('day', s.created_at)::date AS day_start
  FROM auth.sessions s WHERE s.user_id IS NOT NULL
),
first_login AS (
  SELECT user_id, MIN(created_at) AS first_login_time,
         DATE_TRUNC('day', MIN(created_at))::date AS cohort_day_start
  FROM events GROUP BY 1
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


-- =============================================================
-- 11. Users Retention - Login Event - Onboarded Only  (ds101 · 2 charts)
-- =============================================================
CREATE OR REPLACE VIEW analytics.retention_login_onboarded_weekly AS
WITH params AS (SELECT 12::int AS max_weeks, 365::int AS onboarding_window_days),
events AS (
  SELECT s.user_id::text AS user_id, s.created_at::timestamptz AS created_at,
         DATE_TRUNC('week', s.created_at)::date AS week_start
  FROM auth.sessions s WHERE s.user_id IS NOT NULL
),
first_login_all AS (
  SELECT user_id, MIN(created_at) AS first_login_time,
         DATE_TRUNC('week', MIN(created_at))::date AS cohort_week_start
  FROM events GROUP BY 1
),
onboarders AS (
  SELECT fl.user_id FROM first_login_all fl
  WHERE EXISTS (
    SELECT 1 FROM platform."AgentGraphExecution" e
    WHERE e."userId"::text = fl.user_id
      AND e."createdAt" >= fl.first_login_time
      AND e."createdAt" < fl.first_login_time
          + make_interval(days => (SELECT onboarding_window_days FROM params))
  )
),
first_login AS (SELECT * FROM first_login_all WHERE user_id IN (SELECT user_id FROM onboarders)),
activity_weeks AS (SELECT DISTINCT user_id, week_start FROM events),
user_week_age AS (
  SELECT aw.user_id, fl.cohort_week_start,
         ((aw.week_start - DATE_TRUNC('week',fl.first_login_time)::date)/7)::int AS user_lifetime_week
  FROM activity_weeks aw JOIN first_login fl USING (user_id)
  WHERE aw.week_start >= DATE_TRUNC('week',fl.first_login_time)::date
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


-- =============================================================
-- 12. Users Retention - Execution Event (Weekly)  (ds92 · 2 charts)
-- =============================================================
CREATE OR REPLACE VIEW analytics.retention_execution_weekly AS
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


-- =============================================================
-- 13. Users Retention - Execution Event (Daily)  (ds111 · 1 chart)
-- =============================================================
CREATE OR REPLACE VIEW analytics.retention_execution_daily AS
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


-- =============================================================
-- 14. Agent Retention  (ds35 · 2 charts)
-- =============================================================
CREATE OR REPLACE VIEW analytics.retention_agent AS
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
agent_names AS (SELECT g."id" AS agent_id, MAX(g."name") AS agent_name FROM platform."AgentGraph" g GROUP BY 1),
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


-- =============================================================
-- Final: grant SELECT on all analytics views
-- =============================================================
GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO analytics_readonly;
