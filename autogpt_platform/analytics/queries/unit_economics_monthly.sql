-- =============================================================
-- View: analytics.unit_economics_monthly
-- Looker source alias: (new)  |  Charts: 0
-- =============================================================
-- DESCRIPTION
--   One row per (user, calendar month): what the user did, what it
--   cost us in real provider spend, and what we charged them. Built
--   for pricing questions ("what does a free trial with a card on
--   file cost us per user per month?") and margin questions.
--
--   - tasks_human        human-started agent runs + human chat turns
--   - tasks_automated    schedule / webhook runs + scheduled follow-up turns
--   - platform_cost_usd  our provider cost (PlatformCostLog), split into
--                        agent / copilot / background (dream passes)
--   - credits_spent_usd  what we charged in credits (USAGE)
--   - gross_margin_usd   credits_spent_usd - platform_cost_usd (approximate:
--                        credits are prepaid, cost is incurred at use)
--   - cost_per_task_usd  platform_cost_usd / (tasks_human + tasks_automated)
--
--   subscription_tier is the user's CURRENT tier, not a snapshot
--   (there is no tier history table). Sum across users for platform
--   totals; average for "per user".
--
-- SOURCE TABLES
--   platform.AgentGraphExecution, platform.ChatMessage/ChatSession,
--   platform.PlatformCostLog, platform.CreditTransaction, platform.User
--
-- OUTPUT COLUMNS
--   user_id, email, subscription_tier, month (DATE, first of month),
--   tasks_human, tasks_automated, agent_runs, autopilot_turns, expert_turns,
--   active_days, platform_cost_usd, agent_cost_usd, copilot_cost_usd,
--   background_cost_usd, credits_spent_usd, credits_purchased_usd,
--   gross_margin_usd, cost_per_task_usd, cost_per_active_day_usd
--
-- WINDOW
--   Rolling 12 months
--
-- EXAMPLE QUERIES
--   -- Average cost to us per active user per month, by tier
--   SELECT month, subscription_tier,
--          AVG(platform_cost_usd) AS avg_cost_usd,
--          PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY platform_cost_usd) AS p90_cost_usd,
--          COUNT(*) AS users
--   FROM analytics.unit_economics_monthly
--   WHERE tasks_human + tasks_automated > 0
--   GROUP BY 1, 2 ORDER BY 1, 2;
--
--   -- Platform-wide margin per month
--   SELECT month, SUM(credits_spent_usd) AS charged, SUM(platform_cost_usd) AS cost,
--          SUM(gross_margin_usd) AS margin
--   FROM analytics.unit_economics_monthly GROUP BY 1 ORDER BY 1;
-- =============================================================

WITH runs AS (
  SELECT
    ge."userId"                                                          AS user_id,
    DATE_TRUNC('month', ge."createdAt")::date                            AS month,
    COUNT(*)                                                             AS agent_runs,
    COUNT(*) FILTER (WHERE ge."triggerSource" IS NULL
                        OR ge."triggerSource" IN ('manual', 'api', 'copilot'))
                                                                         AS agent_runs_human,
    COUNT(*) FILTER (WHERE ge."triggerSource" IN ('schedule', 'webhook'))
                                                                         AS agent_runs_automated,
    COUNT(DISTINCT ge."createdAt"::date)                                 AS run_days
  FROM platform."AgentGraphExecution" ge
  WHERE ge."createdAt" > DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '11 months'
    AND ge."isDeleted" = FALSE
    AND ge."parentGraphExecutionId" IS NULL
    AND COALESCE(ge."stats"::jsonb->>'is_dry_run', 'false') <> 'true'
  GROUP BY 1, 2
),
turns AS (
  SELECT
    s."userId"                                                           AS user_id,
    DATE_TRUNC('month', m."createdAt")::date                             AS month,
    COUNT(*) FILTER (WHERE s."expertId" IS NULL
                       AND COALESCE(s."metadata"::jsonb->>'origin', 'interactive') <> 'automation')
                                                                         AS autopilot_turns,
    COUNT(*) FILTER (WHERE s."expertId" IS NOT NULL
                       AND COALESCE(s."metadata"::jsonb->>'origin', 'interactive') <> 'automation')
                                                                         AS expert_turns,
    COUNT(*) FILTER (WHERE COALESCE(s."metadata"::jsonb->>'origin', 'interactive') = 'automation')
                                                                         AS scheduled_turns,
    COUNT(DISTINCT m."createdAt"::date)                                  AS turn_days
  FROM platform."ChatMessage" m
  JOIN platform."ChatSession" s ON s."id" = m."sessionId"
  WHERE m."role" = 'user'
    AND m."createdAt" > DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '11 months'
    AND COALESCE(s."metadata"::jsonb->>'kind', 'normal') <> 'dream'
  GROUP BY 1, 2
),
costs AS (
  SELECT
    "userId"                                                             AS user_id,
    DATE_TRUNC('month', "createdAt")::date                               AS month,
    SUM(COALESCE("costMicrodollars", 0)) / 1000000.0                     AS platform_cost_usd,
    SUM(COALESCE("costMicrodollars", 0)) FILTER (
      WHERE COALESCE("blockName", '') NOT ILIKE 'copilot:%') / 1000000.0 AS agent_cost_usd,
    SUM(COALESCE("costMicrodollars", 0)) FILTER (
      WHERE "blockName" ILIKE 'copilot:%'
        AND COALESCE("metadata"::jsonb->>'source', 'copilot') <> 'dream_pass'
        AND "blockName" NOT ILIKE 'copilot:dream%') / 1000000.0          AS copilot_cost_usd,
    SUM(COALESCE("costMicrodollars", 0)) FILTER (
      WHERE "metadata"::jsonb->>'source' = 'dream_pass'
         OR "blockName" ILIKE 'copilot:dream%') / 1000000.0              AS background_cost_usd
  FROM platform."PlatformCostLog"
  WHERE "userId" IS NOT NULL
    AND "createdAt" > DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '11 months'
  GROUP BY 1, 2
),
credits AS (
  SELECT
    "userId"                                                             AS user_id,
    DATE_TRUNC('month', "createdAt")::date                               AS month,
    -COALESCE(SUM("amount") FILTER (WHERE "type" = 'USAGE'), 0) / 100.0  AS credits_spent_usd,
    COALESCE(SUM("amount") FILTER (WHERE "type" IN ('TOP_UP', 'SUBSCRIPTION')), 0) / 100.0
                                                                         AS credits_purchased_usd
  FROM platform."CreditTransaction"
  WHERE "isActive" = TRUE
    AND "createdAt" > DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '11 months'
  GROUP BY 1, 2
),
keys AS (
  SELECT user_id, month FROM runs
  UNION SELECT user_id, month FROM turns
  UNION SELECT user_id, month FROM costs
  UNION SELECT user_id, month FROM credits
),
assembled AS (
  SELECT
    k.user_id,
    u."email"                                                            AS email,
    u."subscriptionTier"::text                                           AS subscription_tier,
    k.month,
    COALESCE(r.agent_runs_human, 0) + COALESCE(t.autopilot_turns, 0) + COALESCE(t.expert_turns, 0)
                                                                         AS tasks_human,
    COALESCE(r.agent_runs_automated, 0) + COALESCE(t.scheduled_turns, 0) AS tasks_automated,
    COALESCE(r.agent_runs, 0)                                            AS agent_runs,
    COALESCE(t.autopilot_turns, 0)                                       AS autopilot_turns,
    COALESCE(t.expert_turns, 0)                                          AS expert_turns,
    GREATEST(COALESCE(r.run_days, 0), COALESCE(t.turn_days, 0))          AS active_days,
    COALESCE(c.platform_cost_usd, 0)                                     AS platform_cost_usd,
    COALESCE(c.agent_cost_usd, 0)                                        AS agent_cost_usd,
    COALESCE(c.copilot_cost_usd, 0)                                      AS copilot_cost_usd,
    COALESCE(c.background_cost_usd, 0)                                   AS background_cost_usd,
    COALESCE(cr.credits_spent_usd, 0)                                    AS credits_spent_usd,
    COALESCE(cr.credits_purchased_usd, 0)                                AS credits_purchased_usd
  FROM keys k
  LEFT JOIN platform."User" u ON u."id" = k.user_id
  LEFT JOIN runs    r  ON r.user_id  = k.user_id AND r.month  = k.month
  LEFT JOIN turns   t  ON t.user_id  = k.user_id AND t.month  = k.month
  LEFT JOIN costs   c  ON c.user_id  = k.user_id AND c.month  = k.month
  LEFT JOIN credits cr ON cr.user_id = k.user_id AND cr.month = k.month
)
SELECT
  a.*,
  a.credits_spent_usd - a.platform_cost_usd                              AS gross_margin_usd,
  a.platform_cost_usd / NULLIF(a.tasks_human + a.tasks_automated, 0)     AS cost_per_task_usd,
  a.platform_cost_usd / NULLIF(a.active_days, 0)                         AS cost_per_active_day_usd
FROM assembled a
