-- =============================================================
-- View: analytics.user_lifecycle
-- Looker source alias: (new)  |  Charts: 0
-- =============================================================
-- DESCRIPTION
--   One row per user with the facts that describe where they are in
--   their lifecycle: when they signed up, when they first and last
--   did anything, how much they did in their first two weeks, what
--   they cost us, what they paid, and the derived labels GTM asked
--   for (activated, stale, churned). It is deliberately wide so it
--   can be used as the feature table for a churn-signal analysis:
--   pick a label column, regress on the rest.
--
--   Definitions (change here, everywhere downstream follows):
--   - task            human-started agent run (manual / API / copilot;
--                     untagged legacy rows count) or a human chat turn
--   - activated       >= 3 tasks on >= 2 distinct days within 14 days of signup
--   - last_active_at  latest of: last task, last visit, last scheduled run,
--                     falling back to signup
--   - stale_14d/30d   no activity in the last 14 / 30 days
--   - churned_30d     had at least one task ever, signed up > 30 days ago,
--                     and no activity in the last 30 days
--   - never_activated_30d  signed up > 30 days ago and never did a task
--
-- SOURCE TABLES
--   platform.User, platform.UserOnboarding, auth.sessions,
--   platform.AgentGraphExecution, platform.ChatMessage/ChatSession,
--   platform.ActivityEvent (schedules), platform.Expert,
--   platform.PlatformCostLog, platform.CreditTransaction
--
-- OUTPUT COLUMNS
--   Identity: user_id, email, signup_at, subscription_tier, timezone,
--             usage_reason, onboarding_completed, onboarding_integrations_selected,
--             integrations_connected_total, integration_providers_connected,
--             first_integration_connected_at
--   Timestamps: first_login_at, last_login_at, last_visit_at, first_agent_run_at,
--             last_agent_run_at, first_chat_turn_at, last_chat_turn_at,
--             first_task_at, last_task_at, last_scheduled_run_at, last_active_at,
--             first_schedule_created_at, first_expert_hired_at, first_purchase_at
--   Counts:   login_count, agent_runs_total, agent_runs_human_total,
--             agent_runs_scheduled_total, agent_runs_failed_total,
--             agent_runs_no_credits_total, expert_workflow_runs_total,
--             distinct_agents_run, scheduled_runs_30d, autopilot_turns_total,
--             expert_turns_total, chat_sessions_total, active_days_total,
--             tasks_first_7d, tasks_first_14d, active_days_first_14d,
--             tasks_7d, tasks_28d, active_days_28d, schedules_created_total,
--             experts_hired_total, experts_active, purchases_total
--   Money:    platform_cost_usd_total, platform_cost_usd_30d,
--             credits_spent_usd_total, credits_purchased_usd_total
--   Derived:  hours_to_first_task, days_since_last_active,
--             first_task_within_7d, activated, stale_14d, stale_30d,
--             churned_30d, never_activated_30d
--
-- WINDOW
--   All users, full history (per-user aggregates)
--
-- EXAMPLE QUERIES
--   -- Activation rate by signup week
--   SELECT DATE_TRUNC('week', signup_at)::date AS week,
--          AVG(CASE WHEN activated THEN 1 ELSE 0 END) AS activation_rate, COUNT(*) AS signups
--   FROM analytics.user_lifecycle
--   WHERE signup_at < NOW() - INTERVAL '14 days'
--   GROUP BY 1 ORDER BY 1;
--
--   -- Which early behaviours separate churned from retained users
--   SELECT churned_30d,
--          AVG(tasks_first_7d) AS avg_tasks_first_7d,
--          AVG(CASE WHEN schedules_created_total > 0 THEN 1 ELSE 0 END) AS pct_with_schedule,
--          AVG(CASE WHEN expert_turns_total > 0 THEN 1 ELSE 0 END) AS pct_used_expert,
--          AVG(agent_runs_failed_total::float / NULLIF(agent_runs_total, 0)) AS avg_fail_ratio
--   FROM analytics.user_lifecycle
--   WHERE first_task_at IS NOT NULL AND signup_at < NOW() - INTERVAL '60 days'
--   GROUP BY 1;
--
--   -- Cost to us per user per month of life (users active this month)
--   SELECT AVG(platform_cost_usd_30d) FROM analytics.user_lifecycle WHERE NOT stale_30d;
-- =============================================================

WITH users AS (
  SELECT
    u."id"                        AS user_id,
    u."email"                     AS email,
    u."createdAt"                 AS signup_at,
    u."subscriptionTier"::text    AS subscription_tier,
    u."timezone"                  AS timezone
  FROM platform."User" u
),
logins AS (
  SELECT
    user_id::text                                                        AS user_id,
    MIN(created_at)                                                      AS first_login_at,
    MAX(created_at)                                                      AS last_login_at,
    GREATEST(MAX(refreshed_at)::timestamptz, MAX(created_at)::timestamptz) AS last_visit_at,
    COUNT(*)                                                             AS login_count
  FROM auth.sessions
  WHERE user_id IS NOT NULL
  GROUP BY 1
),
runs AS (
  SELECT
    ge."userId"                                                          AS user_id,
    MIN(ge."createdAt")                                                  AS first_agent_run_at,
    MAX(ge."createdAt")                                                  AS last_agent_run_at,
    COUNT(*)                                                             AS agent_runs_total,
    COUNT(*) FILTER (WHERE ge."triggerSource" IS NULL
                        OR ge."triggerSource" IN ('manual', 'api', 'copilot'))
                                                                         AS agent_runs_human_total,
    COUNT(*) FILTER (WHERE ge."triggerSource" = 'schedule')              AS agent_runs_scheduled_total,
    COUNT(*) FILTER (WHERE ge."executionStatus" = 'FAILED')              AS agent_runs_failed_total,
    COUNT(*) FILTER (WHERE ge."executionStatus" = 'FAILED'
                       AND ge."stats"::jsonb->>'failure_reason' = 'insufficient_balance')
                                                                         AS agent_runs_no_credits_total,
    COUNT(*) FILTER (WHERE ge."expertId" IS NOT NULL)                    AS expert_workflow_runs_total,
    COUNT(DISTINCT ge."agentGraphId")                                    AS distinct_agents_run,
    MAX(ge."createdAt") FILTER (WHERE ge."triggerSource" = 'schedule')   AS last_scheduled_run_at,
    COUNT(*) FILTER (WHERE ge."triggerSource" = 'schedule'
                       AND ge."createdAt" > NOW() - INTERVAL '30 days')  AS scheduled_runs_30d
  FROM platform."AgentGraphExecution" ge
  WHERE ge."isDeleted" = FALSE
    AND ge."parentGraphExecutionId" IS NULL
    AND COALESCE(ge."stats"::jsonb->>'is_dry_run', 'false') <> 'true'
  GROUP BY 1
),
turns AS (
  SELECT
    s."userId"                                                           AS user_id,
    MIN(m."createdAt")                                                   AS first_chat_turn_at,
    MAX(m."createdAt")                                                   AS last_chat_turn_at,
    COUNT(*) FILTER (WHERE s."expertId" IS NULL)                         AS autopilot_turns_total,
    COUNT(*) FILTER (WHERE s."expertId" IS NOT NULL)                     AS expert_turns_total,
    COUNT(DISTINCT m."sessionId")                                        AS chat_sessions_total
  FROM platform."ChatMessage" m
  JOIN platform."ChatSession" s ON s."id" = m."sessionId"
  WHERE m."role" = 'user'
    AND COALESCE(s."metadata"::jsonb->>'kind', 'normal') <> 'dream'
    AND COALESCE(s."metadata"::jsonb->>'origin', 'interactive') <> 'automation'
  GROUP BY 1
),
task_events AS (
  SELECT ge."userId" AS user_id, ge."createdAt" AS at
  FROM platform."AgentGraphExecution" ge
  WHERE ge."isDeleted" = FALSE
    AND ge."parentGraphExecutionId" IS NULL
    AND COALESCE(ge."stats"::jsonb->>'is_dry_run', 'false') <> 'true'
    AND (ge."triggerSource" IS NULL OR ge."triggerSource" IN ('manual', 'api', 'copilot'))
  UNION ALL
  SELECT s."userId", m."createdAt"
  FROM platform."ChatMessage" m
  JOIN platform."ChatSession" s ON s."id" = m."sessionId"
  WHERE m."role" = 'user'
    AND COALESCE(s."metadata"::jsonb->>'kind', 'normal') <> 'dream'
    AND COALESCE(s."metadata"::jsonb->>'origin', 'interactive') <> 'automation'
),
task_days AS (
  SELECT
    t.user_id,
    COUNT(DISTINCT t.at::date)                                           AS active_days_total,
    COUNT(*) FILTER (WHERE t.at < u.signup_at + INTERVAL '7 days')       AS tasks_first_7d,
    COUNT(*) FILTER (WHERE t.at < u.signup_at + INTERVAL '14 days')      AS tasks_first_14d,
    COUNT(DISTINCT t.at::date) FILTER (WHERE t.at < u.signup_at + INTERVAL '14 days')
                                                                         AS active_days_first_14d,
    COUNT(*) FILTER (WHERE t.at > NOW() - INTERVAL '7 days')             AS tasks_7d,
    COUNT(*) FILTER (WHERE t.at > NOW() - INTERVAL '28 days')            AS tasks_28d,
    COUNT(DISTINCT t.at::date) FILTER (WHERE t.at > NOW() - INTERVAL '28 days')
                                                                         AS active_days_28d
  FROM task_events t
  JOIN users u ON u.user_id = t.user_id
  GROUP BY 1
),
schedules AS (
  SELECT
    "userId"                                                             AS user_id,
    MIN("createdAt") FILTER (WHERE "eventType" = 'schedule.created')     AS first_schedule_created_at,
    COUNT(*) FILTER (WHERE "eventType" = 'schedule.created')             AS schedules_created_total
  FROM platform."ActivityEvent"
  WHERE "category" = 'SCHEDULE'
  GROUP BY 1
),
experts AS (
  SELECT
    "ownerUserId"                                                        AS user_id,
    COUNT(*)                                                             AS experts_hired_total,
    COUNT(*) FILTER (WHERE "isArchived" = FALSE)                         AS experts_active,
    MIN("createdAt")                                                     AS first_expert_hired_at
  FROM platform."Expert"
  WHERE "ownerUserId" IS NOT NULL AND "isTemplate" = FALSE
  GROUP BY 1
),
costs AS (
  SELECT
    "userId"                                                             AS user_id,
    SUM(COALESCE("costMicrodollars", 0)) / 1000000.0                     AS platform_cost_usd_total,
    SUM(COALESCE("costMicrodollars", 0)) FILTER (WHERE "createdAt" > NOW() - INTERVAL '30 days') / 1000000.0
                                                                         AS platform_cost_usd_30d
  FROM platform."PlatformCostLog"
  WHERE "userId" IS NOT NULL
  GROUP BY 1
),
credits AS (
  SELECT
    "userId"                                                             AS user_id,
    -COALESCE(SUM("amount") FILTER (WHERE "type" = 'USAGE'), 0) / 100.0  AS credits_spent_usd_total,
    COALESCE(SUM("amount") FILTER (WHERE "type" IN ('TOP_UP', 'SUBSCRIPTION')), 0) / 100.0
                                                                         AS credits_purchased_usd_total,
    COUNT(*) FILTER (WHERE "type" IN ('TOP_UP', 'SUBSCRIPTION'))         AS purchases_total,
    MIN("createdAt") FILTER (WHERE "type" IN ('TOP_UP', 'SUBSCRIPTION')) AS first_purchase_at
  FROM platform."CreditTransaction"
  WHERE "isActive" = TRUE
  GROUP BY 1
),
onboarding AS (
  SELECT
    "userId"                                                             AS user_id,
    "usageReason"                                                        AS usage_reason,
    'ONBOARDING_COMPLETE' = ANY("completedSteps")                        AS onboarding_completed,
    COALESCE(cardinality("integrations"), 0)                             AS onboarding_integrations_selected
  FROM platform."UserOnboarding"
),
integrations AS (
  SELECT
    "createdByUserId"                                                    AS user_id,
    COUNT(*)                                                             AS integrations_connected_total,
    COUNT(DISTINCT "provider")                                           AS integration_providers_connected,
    MIN("createdAt")                                                     AS first_integration_connected_at
  FROM platform."IntegrationCredential"
  GROUP BY 1
),
assembled AS (
  SELECT
    u.user_id, u.email, u.signup_at, u.subscription_tier, u.timezone,
    o.usage_reason,
    COALESCE(o.onboarding_completed, FALSE)                              AS onboarding_completed,
    COALESCE(o.onboarding_integrations_selected, 0)                      AS onboarding_integrations_selected,
    COALESCE(i.integrations_connected_total, 0)                          AS integrations_connected_total,
    COALESCE(i.integration_providers_connected, 0)                       AS integration_providers_connected,
    i.first_integration_connected_at,
    l.first_login_at, l.last_login_at, l.last_visit_at,
    COALESCE(l.login_count, 0)                                           AS login_count,
    r.first_agent_run_at, r.last_agent_run_at,
    t.first_chat_turn_at, t.last_chat_turn_at,
    LEAST(r.first_agent_run_at, t.first_chat_turn_at)                    AS first_task_at,
    GREATEST(r.last_agent_run_at, t.last_chat_turn_at)                   AS last_task_at,
    r.last_scheduled_run_at,
    COALESCE(
      GREATEST(r.last_agent_run_at, t.last_chat_turn_at, l.last_visit_at, r.last_scheduled_run_at),
      u.signup_at
    )                                                                    AS last_active_at,
    s.first_schedule_created_at,
    e.first_expert_hired_at,
    cr.first_purchase_at,
    COALESCE(r.agent_runs_total, 0)                                      AS agent_runs_total,
    COALESCE(r.agent_runs_human_total, 0)                                AS agent_runs_human_total,
    COALESCE(r.agent_runs_scheduled_total, 0)                            AS agent_runs_scheduled_total,
    COALESCE(r.agent_runs_failed_total, 0)                               AS agent_runs_failed_total,
    COALESCE(r.agent_runs_no_credits_total, 0)                           AS agent_runs_no_credits_total,
    COALESCE(r.expert_workflow_runs_total, 0)                            AS expert_workflow_runs_total,
    COALESCE(r.distinct_agents_run, 0)                                   AS distinct_agents_run,
    COALESCE(r.scheduled_runs_30d, 0)                                    AS scheduled_runs_30d,
    COALESCE(t.autopilot_turns_total, 0)                                 AS autopilot_turns_total,
    COALESCE(t.expert_turns_total, 0)                                    AS expert_turns_total,
    COALESCE(t.chat_sessions_total, 0)                                   AS chat_sessions_total,
    COALESCE(td.active_days_total, 0)                                    AS active_days_total,
    COALESCE(td.tasks_first_7d, 0)                                       AS tasks_first_7d,
    COALESCE(td.tasks_first_14d, 0)                                      AS tasks_first_14d,
    COALESCE(td.active_days_first_14d, 0)                                AS active_days_first_14d,
    COALESCE(td.tasks_7d, 0)                                             AS tasks_7d,
    COALESCE(td.tasks_28d, 0)                                            AS tasks_28d,
    COALESCE(td.active_days_28d, 0)                                      AS active_days_28d,
    COALESCE(s.schedules_created_total, 0)                               AS schedules_created_total,
    COALESCE(e.experts_hired_total, 0)                                   AS experts_hired_total,
    COALESCE(e.experts_active, 0)                                        AS experts_active,
    COALESCE(cr.purchases_total, 0)                                      AS purchases_total,
    COALESCE(c.platform_cost_usd_total, 0)                               AS platform_cost_usd_total,
    COALESCE(c.platform_cost_usd_30d, 0)                                 AS platform_cost_usd_30d,
    COALESCE(cr.credits_spent_usd_total, 0)                              AS credits_spent_usd_total,
    COALESCE(cr.credits_purchased_usd_total, 0)                          AS credits_purchased_usd_total
  FROM users u
  LEFT JOIN onboarding   o  ON o.user_id  = u.user_id
  LEFT JOIN integrations i  ON i.user_id  = u.user_id
  LEFT JOIN logins       l  ON l.user_id  = u.user_id
  LEFT JOIN runs       r  ON r.user_id  = u.user_id
  LEFT JOIN turns      t  ON t.user_id  = u.user_id
  LEFT JOIN task_days  td ON td.user_id = u.user_id
  LEFT JOIN schedules  s  ON s.user_id  = u.user_id
  LEFT JOIN experts    e  ON e.user_id  = u.user_id
  LEFT JOIN costs      c  ON c.user_id  = u.user_id
  LEFT JOIN credits    cr ON cr.user_id = u.user_id
)
SELECT
  a.*,
  EXTRACT(EPOCH FROM (a.first_task_at - a.signup_at)) / 3600.0          AS hours_to_first_task,
  (CURRENT_DATE - a.last_active_at::date)                                AS days_since_last_active,
  a.tasks_first_7d >= 1                                                  AS first_task_within_7d,
  a.tasks_first_14d >= 3 AND a.active_days_first_14d >= 2                AS activated,
  a.last_active_at < NOW() - INTERVAL '14 days'                          AS stale_14d,
  a.last_active_at < NOW() - INTERVAL '30 days'                          AS stale_30d,
  a.first_task_at IS NOT NULL
    AND a.signup_at < NOW() - INTERVAL '30 days'
    AND a.last_active_at < NOW() - INTERVAL '30 days'                    AS churned_30d,
  a.first_task_at IS NULL
    AND a.signup_at < NOW() - INTERVAL '30 days'                         AS never_activated_30d
FROM assembled a
