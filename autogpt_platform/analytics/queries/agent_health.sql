-- =============================================================
-- View: analytics.agent_health
-- Looker source alias: (new)  |  Charts: 0
-- =============================================================
-- DESCRIPTION
--   One row per agent a user has in their library (not deleted or
--   archived): is it running, is it idle, is it failing. This is
--   where "agent_idle" and "agent_fail" come from. An agent that a
--   user added but never ran, or that stopped running, is the
--   earliest churn signal we have.
--
-- SOURCE TABLES
--   platform.LibraryAgent         — the user's agents (lastRunAt)
--   platform.AgentGraph           — agent name
--   platform.AgentGraphExecution  — run outcomes (root, non-dry runs)
--   platform.ActivityEvent        — schedule.created per graph (objectId)
--
-- OUTPUT COLUMNS
--   user_id                  TEXT         Owner
--   graph_id                 TEXT         Agent graph UUID
--   library_agent_id         TEXT         LibraryAgent UUID
--   agent_name               TEXT         Display name
--   is_created_by_user       BOOLEAN      Built by the user (vs. added from marketplace)
--   added_at                 TIMESTAMPTZ  When it entered the library
--   last_run_at              TIMESTAMPTZ  Last run of any kind (nullable = never run)
--   last_completed_at        TIMESTAMPTZ  Last successful run
--   last_failed_at           TIMESTAMPTZ  Last failed run
--   runs_total               BIGINT       Lifetime root runs
--   runs_7d / runs_30d       BIGINT       Runs in the last 7 / 30 days
--   completed_30d            BIGINT       COMPLETED runs, last 30 days
--   failed_30d               BIGINT       FAILED runs, last 30 days
--   scheduled_runs_30d       BIGINT       Runs fired by a schedule, last 30 days
--   webhook_runs_30d         BIGINT       Runs fired by a webhook, last 30 days
--   fail_rate_30d            FLOAT        failed / (completed + failed), last 30 days
--   schedules_created_total  BIGINT       schedule.created events for this graph
--   days_since_last_run      INT          NULL if never run
--   never_run                BOOLEAN      Added but never executed
--   idle_7d / idle_30d       BOOLEAN      Has run before, but not in the last 7 / 30 days
--   failing                  BOOLEAN      >= 3 failures and >= 50% fail rate in 30 days
--
-- WINDOW
--   All library agents; run windows are rolling 7 / 30 days
--
-- EXAMPLE QUERIES
--   -- Users with at least one agent that went idle this month
--   SELECT COUNT(DISTINCT user_id) FROM analytics.agent_health WHERE idle_30d;
--
--   -- Share of library agents that never ran
--   SELECT AVG(CASE WHEN never_run THEN 1 ELSE 0 END) FROM analytics.agent_health;
--
--   -- Failing agents to reach out about
--   SELECT user_id, agent_name, failed_30d, fail_rate_30d, last_failed_at
--   FROM analytics.agent_health WHERE failing ORDER BY failed_30d DESC;
-- =============================================================

WITH agents AS (
  SELECT
    la."userId"                              AS user_id,
    la."agentGraphId"                        AS graph_id,
    la."id"                                  AS library_agent_id,
    COALESCE(la."name", g."name")            AS agent_name,
    la."isCreatedByUser"                     AS is_created_by_user,
    la."createdAt"                           AS added_at,
    la."lastRunAt"                           AS last_run_at
  FROM platform."LibraryAgent" la
  LEFT JOIN platform."AgentGraph" g
         ON g."id" = la."agentGraphId" AND g."version" = la."agentGraphVersion"
  WHERE la."isDeleted" = FALSE
    AND la."isArchived" = FALSE
),
runs AS (
  SELECT
    ge."userId"                                                        AS user_id,
    ge."agentGraphId"                                                  AS graph_id,
    COUNT(*)                                                           AS runs_total,
    COUNT(*) FILTER (WHERE ge."createdAt" > NOW() - INTERVAL '7 days')  AS runs_7d,
    COUNT(*) FILTER (WHERE ge."createdAt" > NOW() - INTERVAL '30 days') AS runs_30d,
    COUNT(*) FILTER (WHERE ge."createdAt" > NOW() - INTERVAL '30 days'
                       AND ge."executionStatus" = 'COMPLETED')          AS completed_30d,
    COUNT(*) FILTER (WHERE ge."createdAt" > NOW() - INTERVAL '30 days'
                       AND ge."executionStatus" = 'FAILED')             AS failed_30d,
    COUNT(*) FILTER (WHERE ge."createdAt" > NOW() - INTERVAL '30 days'
                       AND ge."triggerSource" = 'schedule')             AS scheduled_runs_30d,
    COUNT(*) FILTER (WHERE ge."createdAt" > NOW() - INTERVAL '30 days'
                       AND ge."triggerSource" = 'webhook')              AS webhook_runs_30d,
    MAX(ge."createdAt") FILTER (WHERE ge."executionStatus" = 'COMPLETED') AS last_completed_at,
    MAX(ge."createdAt") FILTER (WHERE ge."executionStatus" = 'FAILED')    AS last_failed_at
  FROM platform."AgentGraphExecution" ge
  WHERE ge."isDeleted" = FALSE
    AND ge."parentGraphExecutionId" IS NULL
    AND COALESCE(ge."stats"::jsonb->>'is_dry_run', 'false') <> 'true'
  GROUP BY 1, 2
),
schedules AS (
  SELECT
    "userId"   AS user_id,
    "objectId" AS graph_id,
    COUNT(*) FILTER (WHERE "eventType" = 'schedule.created') AS schedules_created_total
  FROM platform."ActivityEvent"
  WHERE "category" = 'SCHEDULE' AND "objectId" IS NOT NULL
  GROUP BY 1, 2
)
SELECT
  a.user_id,
  a.graph_id,
  a.library_agent_id,
  a.agent_name,
  a.is_created_by_user,
  a.added_at,
  a.last_run_at,
  r.last_completed_at,
  r.last_failed_at,
  COALESCE(r.runs_total, 0)                    AS runs_total,
  COALESCE(r.runs_7d, 0)                       AS runs_7d,
  COALESCE(r.runs_30d, 0)                      AS runs_30d,
  COALESCE(r.completed_30d, 0)                 AS completed_30d,
  COALESCE(r.failed_30d, 0)                    AS failed_30d,
  COALESCE(r.scheduled_runs_30d, 0)            AS scheduled_runs_30d,
  COALESCE(r.webhook_runs_30d, 0)              AS webhook_runs_30d,
  COALESCE(r.failed_30d, 0)::float
    / NULLIF(COALESCE(r.completed_30d, 0) + COALESCE(r.failed_30d, 0), 0)
                                               AS fail_rate_30d,
  COALESCE(s.schedules_created_total, 0)       AS schedules_created_total,
  (CURRENT_DATE - a.last_run_at::date)         AS days_since_last_run,
  a.last_run_at IS NULL                        AS never_run,
  a.last_run_at IS NOT NULL AND a.last_run_at < NOW() - INTERVAL '7 days'  AS idle_7d,
  a.last_run_at IS NOT NULL AND a.last_run_at < NOW() - INTERVAL '30 days' AS idle_30d,
  COALESCE(r.failed_30d, 0) >= 3
    AND COALESCE(r.failed_30d, 0)::float
        / NULLIF(COALESCE(r.completed_30d, 0) + COALESCE(r.failed_30d, 0), 0) >= 0.5
                                               AS failing
FROM agents a
LEFT JOIN runs      r ON r.user_id = a.user_id AND r.graph_id = a.graph_id
LEFT JOIN schedules s ON s.user_id = a.user_id AND s.graph_id = a.graph_id
