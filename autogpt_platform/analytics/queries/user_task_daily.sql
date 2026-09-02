-- =============================================================
-- View: analytics.user_task_daily
-- Looker source alias: (new)  |  Charts: 0
-- =============================================================
-- DESCRIPTION
--   One row per (user, day) with everything the user did and cost
--   that day, across every surface: classic agent runs (split by how
--   they were started), Autopilot turns, expert turns and expert
--   workflow runs, schedules created, our real provider cost, and the
--   credits we charged. This is the workhorse table for "how many
--   tasks are people running" and for per-day cost per user.
--
--   A "task" is a unit of work a person asked for now: a human-started
--   agent run (manual UI, API key or copilot tool) or a human chat
--   turn. Automated work (schedule fires, webhook triggers, scheduled
--   follow-ups) is counted separately so the two can be compared.
--
--   Sub-graph runs (parentGraphExecutionId set) and dry runs are
--   excluded from run counts: they are part of the parent task.
--   Runs created before triggerSource existed show up in
--   agent_runs_untagged and cannot be split by start method.
--
-- SOURCE TABLES
--   platform.AgentGraphExecution — agent runs (triggerSource, expertId, stats)
--   platform.ChatMessage / ChatSession — copilot turns
--   platform.ActivityEvent       — schedule.created / schedule.deleted
--   platform.PlatformCostLog     — our provider cost (microdollars)
--   platform.CreditTransaction   — credits charged to / bought by the user (cents)
--   auth.sessions                — logins
--
-- OUTPUT COLUMNS
--   user_id                      TEXT     User UUID
--   day                          DATE     Calendar day (UTC)
--   tasks_human                  BIGINT   agent_runs_human + autopilot_turns + expert_turns
--   tasks_automated              BIGINT   scheduled + webhook runs + scheduled follow-up turns
--   agent_runs                   BIGINT   All root, non-dry agent runs created that day
--   agent_runs_human             BIGINT   triggerSource IN (manual, api, copilot)
--   agent_runs_scheduled         BIGINT   triggerSource = schedule
--   agent_runs_webhook           BIGINT   triggerSource = webhook
--   agent_runs_untagged          BIGINT   triggerSource IS NULL (pre-deploy rows)
--   agent_runs_completed         BIGINT   Terminal status COMPLETED (as of query time)
--   agent_runs_failed            BIGINT   Terminal status FAILED
--   agent_runs_no_credits        BIGINT   FAILED with failure_reason insufficient_balance
--   expert_workflow_runs         BIGINT   Runs attributed to a hired expert
--   autopilot_turns              BIGINT   Human turns in Autopilot chats
--   expert_turns                 BIGINT   Human turns in expert chats
--   scheduled_turns              BIGINT   Model-authored scheduled follow-up turns
--   chat_sessions_touched        BIGINT   Distinct chat sessions with a user turn
--   schedules_created            BIGINT   schedule.created activity events
--   schedules_deleted            BIGINT   schedule.deleted activity events
--   logins                       BIGINT   Supabase sessions created
--   platform_cost_usd            NUMERIC  Our total provider cost for the user that day
--   agent_cost_usd               NUMERIC  ...of which block/agent runs
--   copilot_cost_usd             NUMERIC  ...of which copilot turns (excl. dream passes)
--   background_cost_usd          NUMERIC  ...of which dream/memory passes (no user action)
--   credits_spent_usd            NUMERIC  USAGE transactions (what we charged)
--   credits_purchased_usd        NUMERIC  TOP_UP + SUBSCRIPTION transactions
--   run_credits_usd              NUMERIC  Sum of stats.cost on runs created that day
--
-- WINDOW
--   Rolling 90 days
--
-- EXAMPLE QUERIES
--   -- Tasks per active user per day (platform-wide)
--   SELECT day, SUM(tasks_human)::float / NULLIF(COUNT(*) FILTER (WHERE tasks_human > 0), 0)
--   FROM analytics.user_task_daily GROUP BY 1 ORDER BY 1;
--
--   -- Cost to us per task (human tasks only), last 30 days
--   SELECT SUM(platform_cost_usd) / NULLIF(SUM(tasks_human), 0) AS cost_per_task_usd
--   FROM analytics.user_task_daily WHERE day >= CURRENT_DATE - 30;
--
--   -- Daily active users by surface
--   SELECT day,
--          COUNT(*) FILTER (WHERE agent_runs_human > 0) AS agent_dau,
--          COUNT(*) FILTER (WHERE autopilot_turns > 0) AS autopilot_dau,
--          COUNT(*) FILTER (WHERE expert_turns + expert_workflow_runs > 0) AS expert_dau
--   FROM analytics.user_task_daily GROUP BY 1 ORDER BY 1;
-- =============================================================

WITH runs AS (
  SELECT
    ge."userId"                                          AS user_id,
    DATE_TRUNC('day', ge."createdAt")::date              AS day,
    COUNT(*)                                             AS agent_runs,
    COUNT(*) FILTER (WHERE ge."triggerSource" IN ('manual', 'api', 'copilot'))
                                                         AS agent_runs_human,
    COUNT(*) FILTER (WHERE ge."triggerSource" = 'schedule')
                                                         AS agent_runs_scheduled,
    COUNT(*) FILTER (WHERE ge."triggerSource" = 'webhook')
                                                         AS agent_runs_webhook,
    COUNT(*) FILTER (WHERE ge."triggerSource" IS NULL)   AS agent_runs_untagged,
    COUNT(*) FILTER (WHERE ge."executionStatus" = 'COMPLETED')
                                                         AS agent_runs_completed,
    COUNT(*) FILTER (WHERE ge."executionStatus" = 'FAILED')
                                                         AS agent_runs_failed,
    COUNT(*) FILTER (WHERE ge."executionStatus" = 'FAILED'
                       AND ge."stats"::jsonb->>'failure_reason' = 'insufficient_balance')
                                                         AS agent_runs_no_credits,
    COUNT(*) FILTER (WHERE ge."expertId" IS NOT NULL)    AS expert_workflow_runs,
    COALESCE(SUM((ge."stats"::jsonb->>'cost')::numeric), 0) / 100.0
                                                         AS run_credits_usd
  FROM platform."AgentGraphExecution" ge
  WHERE ge."createdAt" > CURRENT_DATE - INTERVAL '90 days'
    AND ge."isDeleted" = FALSE
    AND ge."parentGraphExecutionId" IS NULL
    AND COALESCE(ge."stats"::jsonb->>'is_dry_run', 'false') <> 'true'
  GROUP BY 1, 2
),
turns AS (
  SELECT
    s."userId"                                           AS user_id,
    DATE_TRUNC('day', m."createdAt")::date               AS day,
    COUNT(*) FILTER (WHERE s."expertId" IS NULL
                       AND COALESCE(s."metadata"::jsonb->>'origin', 'interactive') <> 'automation')
                                                         AS autopilot_turns,
    COUNT(*) FILTER (WHERE s."expertId" IS NOT NULL
                       AND COALESCE(s."metadata"::jsonb->>'origin', 'interactive') <> 'automation')
                                                         AS expert_turns,
    COUNT(*) FILTER (WHERE COALESCE(s."metadata"::jsonb->>'origin', 'interactive') = 'automation')
                                                         AS scheduled_turns,
    COUNT(DISTINCT m."sessionId")                        AS chat_sessions_touched
  FROM platform."ChatMessage" m
  JOIN platform."ChatSession" s ON s."id" = m."sessionId"
  WHERE m."role" = 'user'
    AND m."createdAt" > CURRENT_DATE - INTERVAL '90 days'
    AND COALESCE(s."metadata"::jsonb->>'kind', 'normal') <> 'dream'
  GROUP BY 1, 2
),
schedules AS (
  SELECT
    "userId"                                             AS user_id,
    DATE_TRUNC('day', "createdAt")::date                 AS day,
    COUNT(*) FILTER (WHERE "eventType" = 'schedule.created') AS schedules_created,
    COUNT(*) FILTER (WHERE "eventType" = 'schedule.deleted') AS schedules_deleted
  FROM platform."ActivityEvent"
  WHERE "category" = 'SCHEDULE'
    AND "createdAt" > CURRENT_DATE - INTERVAL '90 days'
  GROUP BY 1, 2
),
costs AS (
  SELECT
    "userId"                                             AS user_id,
    DATE_TRUNC('day', "createdAt")::date                 AS day,
    SUM(COALESCE("costMicrodollars", 0))                 AS platform_cost_md,
    SUM(COALESCE("costMicrodollars", 0)) FILTER (
      WHERE COALESCE("blockName", '') NOT ILIKE 'copilot:%')
                                                         AS agent_cost_md,
    SUM(COALESCE("costMicrodollars", 0)) FILTER (
      WHERE "blockName" ILIKE 'copilot:%'
        AND COALESCE("metadata"::jsonb->>'source', 'copilot') <> 'dream_pass'
        AND "blockName" NOT ILIKE 'copilot:dream%')
                                                         AS copilot_cost_md,
    SUM(COALESCE("costMicrodollars", 0)) FILTER (
      WHERE "metadata"::jsonb->>'source' = 'dream_pass'
         OR "blockName" ILIKE 'copilot:dream%')
                                                         AS background_cost_md
  FROM platform."PlatformCostLog"
  WHERE "userId" IS NOT NULL
    AND "createdAt" > CURRENT_DATE - INTERVAL '90 days'
  GROUP BY 1, 2
),
credits AS (
  SELECT
    "userId"                                             AS user_id,
    DATE_TRUNC('day', "createdAt")::date                 AS day,
    -COALESCE(SUM("amount") FILTER (WHERE "type" = 'USAGE'), 0) / 100.0
                                                         AS credits_spent_usd,
    COALESCE(SUM("amount") FILTER (WHERE "type" IN ('TOP_UP', 'SUBSCRIPTION')), 0) / 100.0
                                                         AS credits_purchased_usd
  FROM platform."CreditTransaction"
  WHERE "isActive" = TRUE
    AND "createdAt" > CURRENT_DATE - INTERVAL '90 days'
  GROUP BY 1, 2
),
logins AS (
  SELECT
    user_id::text                                        AS user_id,
    DATE_TRUNC('day', created_at)::date                  AS day,
    COUNT(*)                                             AS logins
  FROM auth.sessions
  WHERE user_id IS NOT NULL
    AND created_at > CURRENT_DATE - INTERVAL '90 days'
  GROUP BY 1, 2
),
keys AS (
  SELECT user_id, day FROM runs
  UNION SELECT user_id, day FROM turns
  UNION SELECT user_id, day FROM schedules
  UNION SELECT user_id, day FROM costs
  UNION SELECT user_id, day FROM credits
  UNION SELECT user_id, day FROM logins
)
SELECT
  k.user_id,
  k.day,
  COALESCE(r.agent_runs_human, 0) + COALESCE(t.autopilot_turns, 0) + COALESCE(t.expert_turns, 0)
                                                         AS tasks_human,
  COALESCE(r.agent_runs_scheduled, 0) + COALESCE(r.agent_runs_webhook, 0) + COALESCE(t.scheduled_turns, 0)
                                                         AS tasks_automated,
  COALESCE(r.agent_runs, 0)                              AS agent_runs,
  COALESCE(r.agent_runs_human, 0)                        AS agent_runs_human,
  COALESCE(r.agent_runs_scheduled, 0)                    AS agent_runs_scheduled,
  COALESCE(r.agent_runs_webhook, 0)                      AS agent_runs_webhook,
  COALESCE(r.agent_runs_untagged, 0)                     AS agent_runs_untagged,
  COALESCE(r.agent_runs_completed, 0)                    AS agent_runs_completed,
  COALESCE(r.agent_runs_failed, 0)                       AS agent_runs_failed,
  COALESCE(r.agent_runs_no_credits, 0)                   AS agent_runs_no_credits,
  COALESCE(r.expert_workflow_runs, 0)                    AS expert_workflow_runs,
  COALESCE(t.autopilot_turns, 0)                         AS autopilot_turns,
  COALESCE(t.expert_turns, 0)                            AS expert_turns,
  COALESCE(t.scheduled_turns, 0)                         AS scheduled_turns,
  COALESCE(t.chat_sessions_touched, 0)                   AS chat_sessions_touched,
  COALESCE(s.schedules_created, 0)                       AS schedules_created,
  COALESCE(s.schedules_deleted, 0)                       AS schedules_deleted,
  COALESCE(l.logins, 0)                                  AS logins,
  COALESCE(c.platform_cost_md, 0) / 1000000.0            AS platform_cost_usd,
  COALESCE(c.agent_cost_md, 0) / 1000000.0               AS agent_cost_usd,
  COALESCE(c.copilot_cost_md, 0) / 1000000.0             AS copilot_cost_usd,
  COALESCE(c.background_cost_md, 0) / 1000000.0          AS background_cost_usd,
  COALESCE(cr.credits_spent_usd, 0)                      AS credits_spent_usd,
  COALESCE(cr.credits_purchased_usd, 0)                  AS credits_purchased_usd,
  COALESCE(r.run_credits_usd, 0)                         AS run_credits_usd
FROM keys k
LEFT JOIN runs      r  ON r.user_id  = k.user_id AND r.day  = k.day
LEFT JOIN turns     t  ON t.user_id  = k.user_id AND t.day  = k.day
LEFT JOIN schedules s  ON s.user_id  = k.user_id AND s.day  = k.day
LEFT JOIN costs     c  ON c.user_id  = k.user_id AND c.day  = k.day
LEFT JOIN credits   cr ON cr.user_id = k.user_id AND cr.day = k.day
LEFT JOIN logins    l  ON l.user_id  = k.user_id AND l.day  = k.day
