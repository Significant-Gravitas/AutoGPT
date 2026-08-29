-- =============================================================
-- View: analytics.platform_cost_log
-- Looker source alias: ds115  |  Charts: 0
-- =============================================================
-- DESCRIPTION
--   One row per platform cost log entry (last 90 days).
--   Tracks real API spend at the call level: provider, model,
--   token counts (including Anthropic cache tokens), cost in
--   microdollars, and the block/execution that incurred the cost.
--   Joins the User table to provide email for per-user breakdowns.
--
-- SOURCE TABLES
--   platform.PlatformCostLog  — Per-call cost records
--   platform.User             — User email
--
-- OUTPUT COLUMNS
--   id                      TEXT         Log entry UUID
--   createdAt               TIMESTAMPTZ  When the cost was recorded
--   userId                  TEXT         User who incurred the cost (nullable)
--   email                   TEXT         User email (nullable)
--   graphExecId             TEXT         Graph execution UUID (nullable)
--   nodeExecId              TEXT         Node execution UUID (nullable)
--   blockName               TEXT         Block that made the API call (nullable)
--   provider                TEXT         API provider, lowercase (e.g. 'openai', 'anthropic')
--   model                   TEXT         Model name (nullable)
--   trackingType            TEXT         Cost unit: 'tokens' | 'cost_usd' | 'characters' | etc.
--   costMicrodollars        BIGINT       Cost in microdollars (divide by 1,000,000 for USD)
--   costUsd                 FLOAT        Cost in USD (costMicrodollars / 1,000,000)
--   inputTokens             INT          Prompt/input tokens (nullable)
--   outputTokens            INT          Completion/output tokens (nullable)
--   cacheReadTokens         INT          Anthropic cache-read tokens billed at 10% (nullable)
--   cacheCreationTokens     INT          Anthropic cache-write tokens billed at 125% (nullable)
--   totalTokens             INT          inputTokens + outputTokens (nullable if either is null)
--   duration                FLOAT        API call duration in seconds (nullable)
--
-- WINDOW
--   Rolling 90 days (createdAt > CURRENT_DATE - 90 days)
--
-- EXAMPLE QUERIES
--   -- Total spend by provider (last 90 days)
--   SELECT provider, SUM("costUsd") AS total_usd, COUNT(*) AS calls
--   FROM analytics.platform_cost_log
--   GROUP BY 1 ORDER BY total_usd DESC;
--
--   -- Spend by model
--   SELECT provider, model, SUM("costUsd") AS total_usd,
--          SUM("inputTokens") AS input_tokens,
--          SUM("outputTokens") AS output_tokens
--   FROM analytics.platform_cost_log
--   WHERE model IS NOT NULL
--   GROUP BY 1, 2 ORDER BY total_usd DESC;
--
--   -- Top 20 users by spend
--   SELECT "userId", email, SUM("costUsd") AS total_usd, COUNT(*) AS calls
--   FROM analytics.platform_cost_log
--   WHERE "userId" IS NOT NULL
--   GROUP BY 1, 2 ORDER BY total_usd DESC LIMIT 20;
--
--   -- Daily spend trend
--   SELECT DATE_TRUNC('day', "createdAt") AS day,
--          SUM("costUsd") AS daily_usd,
--          COUNT(*) AS calls
--   FROM analytics.platform_cost_log
--   GROUP BY 1 ORDER BY 1;
--
--   -- Cache hit rate for Anthropic (cache reads vs total reads)
--   SELECT DATE_TRUNC('day', "createdAt") AS day,
--          SUM("cacheReadTokens")::float /
--            NULLIF(SUM("inputTokens" + COALESCE("cacheReadTokens", 0)), 0) AS cache_hit_rate
--   FROM analytics.platform_cost_log
--   WHERE provider = 'anthropic'
--   GROUP BY 1 ORDER BY 1;
-- =============================================================

SELECT
    p."id"                                                        AS id,
    p."createdAt"                                                 AS createdAt,
    p."userId"                                                    AS userId,
    u."email"                                                     AS email,
    p."graphExecId"                                               AS graphExecId,
    p."nodeExecId"                                                AS nodeExecId,
    p."blockName"                                                 AS blockName,
    p."provider"                                                  AS provider,
    p."model"                                                     AS model,
    p."trackingType"                                              AS trackingType,
    p."costMicrodollars"                                          AS costMicrodollars,
    p."costMicrodollars"::float / 1000000.0                       AS costUsd,
    p."inputTokens"                                               AS inputTokens,
    p."outputTokens"                                              AS outputTokens,
    p."cacheReadTokens"                                           AS cacheReadTokens,
    p."cacheCreationTokens"                                       AS cacheCreationTokens,
    CASE
        WHEN p."inputTokens" IS NOT NULL AND p."outputTokens" IS NOT NULL
        THEN p."inputTokens" + p."outputTokens"
        ELSE NULL
    END                                                           AS totalTokens,
    p."duration"                                                  AS duration
FROM platform."PlatformCostLog" p
LEFT JOIN platform."User" u ON u."id" = p."userId"
WHERE p."createdAt" > CURRENT_DATE - INTERVAL '90 days'
