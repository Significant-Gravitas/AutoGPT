-- =============================================================
-- View: analytics.users_activities
-- Looker source alias: ds56  |  Charts: 5
-- =============================================================
-- DESCRIPTION
--   One row per user with lifetime activity summary.
--   Joins login sessions with agent graphs, executions and
--   node-level runs to give a full picture of how engaged
--   each user is.  Includes a convenience flag for 7-day
--   activation (did the user return at least 7 days after
--   their first login?).
--
-- SOURCE TABLES
--   auth.sessions                    — Login/session records
--   platform.AgentGraph              — Graphs (agents) built by the user
--   platform.AgentGraphExecution     — Agent run history
--   platform.AgentNodeExecution      — Individual block execution history
--
-- PERFORMANCE NOTE
--   Each CTE aggregates its own table independently by userId.
--   This avoids the fan-out that occurs when driving every join
--   from user_logins across the two largest tables
--   (AgentGraphExecution and AgentNodeExecution).
--
-- OUTPUT COLUMNS
--   user_id                   TEXT         Supabase user UUID
--   first_login_time          TIMESTAMPTZ  First ever session created_at
--   last_login_time           TIMESTAMPTZ  Most recent session created_at
--   last_visit_time           TIMESTAMPTZ  Max of last refresh or login
--   last_agent_save_time      TIMESTAMPTZ  Last time user saved an agent graph
--   agent_count               BIGINT       Number of distinct active graphs built (0 if none)
--   first_agent_run_time      TIMESTAMPTZ  First ever graph execution
--   last_agent_run_time       TIMESTAMPTZ  Most recent graph execution
--   unique_agent_runs         BIGINT       Distinct agent graphs ever run (0 if none)
--   agent_runs                BIGINT       Total graph execution count (0 if none)
--   node_execution_count      BIGINT       Total node executions across all runs
--   node_execution_failed     BIGINT       Node executions with FAILED status
--   node_execution_completed  BIGINT       Node executions with COMPLETED status
--   node_execution_terminated BIGINT       Node executions with TERMINATED status
--   node_execution_queued     BIGINT       Node executions with QUEUED status
--   node_execution_running    BIGINT       Node executions with RUNNING status
--   is_active_after_7d        INT          1=returned after day 7, 0=did not, NULL=too early to tell
--   node_execution_incomplete BIGINT       Node executions with INCOMPLETE status
--   node_execution_review     BIGINT       Node executions with REVIEW status
--
-- EXAMPLE QUERIES
--   -- Users who ran at least one agent and returned after 7 days
--   SELECT COUNT(*) FROM analytics.users_activities
--   WHERE agent_runs > 0 AND is_active_after_7d = 1;
--
--   -- Top 10 most active users by agent runs
--   SELECT user_id, agent_runs, node_execution_count
--   FROM analytics.users_activities
--   ORDER BY agent_runs DESC LIMIT 10;
--
--   -- 7-day activation rate
--   SELECT
--     SUM(CASE WHEN is_active_after_7d = 1 THEN 1 ELSE 0 END)::float
--     / NULLIF(COUNT(CASE WHEN is_active_after_7d IS NOT NULL THEN 1 END), 0)
--     AS activation_rate
--   FROM analytics.users_activities;
-- =============================================================

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
  -- Aggregate AgentGraph directly by userId (no fan-out from user_logins)
  SELECT
    "userId"::text                AS user_id,
    MAX("updatedAt")              AS last_agent_save_time,
    COUNT(DISTINCT "id")          AS agent_count
  FROM platform."AgentGraph"
  WHERE "isActive"
  GROUP BY "userId"
),
user_graph_runs AS (
  -- Aggregate AgentGraphExecution directly by userId
  SELECT
    "userId"::text                        AS user_id,
    MIN("createdAt")                      AS first_agent_run_time,
    MAX("createdAt")                      AS last_agent_run_time,
    COUNT(DISTINCT "agentGraphId")        AS unique_agent_runs,
    COUNT("id")                           AS agent_runs
  FROM platform."AgentGraphExecution"
  GROUP BY "userId"
),
user_node_runs AS (
  -- Aggregate AgentNodeExecution directly; resolve userId via a
  -- single join to AgentGraphExecution instead of fanning out from
  -- user_logins through both large tables.
  SELECT
    g."userId"::text                                                   AS user_id,
    COUNT(*)                                                           AS node_execution_count,
    COUNT(*) FILTER (WHERE n."executionStatus" = 'FAILED')             AS node_execution_failed,
    COUNT(*) FILTER (WHERE n."executionStatus" = 'COMPLETED')          AS node_execution_completed,
    COUNT(*) FILTER (WHERE n."executionStatus" = 'TERMINATED')         AS node_execution_terminated,
    COUNT(*) FILTER (WHERE n."executionStatus" = 'QUEUED')             AS node_execution_queued,
    COUNT(*) FILTER (WHERE n."executionStatus" = 'RUNNING')            AS node_execution_running,
    COUNT(*) FILTER (WHERE n."executionStatus" = 'INCOMPLETE')         AS node_execution_incomplete,
    COUNT(*) FILTER (WHERE n."executionStatus" = 'REVIEW')             AS node_execution_review
  FROM platform."AgentNodeExecution" n
  JOIN platform."AgentGraphExecution" g
    ON g."id" = n."agentGraphExecutionId"
  GROUP BY g."userId"
)
SELECT
  ul.user_id,
  ul.first_login_time,
  ul.last_login_time,
  ul.last_visit_time,
  ua.last_agent_save_time,
  COALESCE(ua.agent_count, 0)             AS agent_count,
  gr.first_agent_run_time,
  gr.last_agent_run_time,
  COALESCE(gr.unique_agent_runs, 0)       AS unique_agent_runs,
  COALESCE(gr.agent_runs, 0)              AS agent_runs,
  COALESCE(nr.node_execution_count, 0)      AS node_execution_count,
  COALESCE(nr.node_execution_failed, 0)     AS node_execution_failed,
  COALESCE(nr.node_execution_completed, 0)  AS node_execution_completed,
  COALESCE(nr.node_execution_terminated, 0) AS node_execution_terminated,
  COALESCE(nr.node_execution_queued, 0)     AS node_execution_queued,
  COALESCE(nr.node_execution_running, 0)    AS node_execution_running,
  CASE
    WHEN ul.first_login_time < NOW() - INTERVAL '7 days'
     AND ul.last_visit_time  >= ul.first_login_time + INTERVAL '7 days' THEN 1
    WHEN ul.first_login_time < NOW() - INTERVAL '7 days'
     AND ul.last_visit_time  <  ul.first_login_time + INTERVAL '7 days' THEN 0
    ELSE NULL
  END AS is_active_after_7d,
  COALESCE(nr.node_execution_incomplete, 0) AS node_execution_incomplete,
  COALESCE(nr.node_execution_review, 0)     AS node_execution_review
FROM user_logins ul
LEFT JOIN user_agents     ua ON ul.user_id = ua.user_id
LEFT JOIN user_graph_runs gr ON ul.user_id = gr.user_id
LEFT JOIN user_node_runs  nr ON ul.user_id = nr.user_id
