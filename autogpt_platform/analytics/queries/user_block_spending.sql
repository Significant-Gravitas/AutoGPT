-- =============================================================
-- View: analytics.user_block_spending
-- Looker source alias: ds6  |  Charts: 5
-- =============================================================
-- DESCRIPTION
--   One row per credit transaction (last 90 days).
--   Shows how users spend credits broken down by block type,
--   LLM provider and model.  Joins node execution stats for
--   token-level detail.
--
-- SOURCE TABLES
--   platform.CreditTransaction   — Credit debit/credit records
--   platform.AgentNodeExecution  — Node execution stats (for token counts)
--
-- OUTPUT COLUMNS
--   transactionKey        TEXT         Unique transaction identifier
--   userId                TEXT         User who was charged
--   amount                DECIMAL      Credit amount (positive = credit, negative = debit)
--   negativeAmount        DECIMAL      amount * -1 (convenience for spend charts)
--   transactionType       TEXT         Transaction type (e.g. 'USAGE', 'REFUND', 'TOP_UP')
--   transactionTime       TIMESTAMPTZ  When the transaction was recorded
--   blockId               TEXT         Block UUID that triggered the spend
--   blockName             TEXT         Human-readable block name
--   llm_provider          TEXT         LLM provider (e.g. 'openai', 'anthropic')
--   llm_model             TEXT         Model name (e.g. 'gpt-4o', 'claude-3-5-sonnet')
--   node_exec_id          TEXT         Linked node execution UUID
--   llm_call_count        INT          LLM API calls made in that execution
--   llm_retry_count       INT          LLM retries in that execution
--   llm_input_token_count INT          Input tokens consumed
--   llm_output_token_count INT         Output tokens produced
--
-- WINDOW
--   Rolling 90 days (createdAt > CURRENT_DATE - 90 days)
--
-- EXAMPLE QUERIES
--   -- Total spend per user (last 90 days)
--   SELECT "userId", SUM("negativeAmount") AS total_spent
--   FROM analytics.user_block_spending
--   WHERE "transactionType" = 'USAGE'
--   GROUP BY 1 ORDER BY total_spent DESC;
--
--   -- Spend by LLM provider + model
--   SELECT "llm_provider", "llm_model",
--          SUM("negativeAmount") AS total_cost,
--          SUM("llm_input_token_count") AS input_tokens,
--          SUM("llm_output_token_count") AS output_tokens
--   FROM analytics.user_block_spending
--   WHERE "llm_provider" IS NOT NULL
--   GROUP BY 1, 2 ORDER BY total_cost DESC;
-- =============================================================

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
    (ne."stats"->>'llm_call_count')::int                       AS llm_call_count,
    (ne."stats"->>'llm_retry_count')::int                      AS llm_retry_count,
    (ne."stats"->>'input_token_count')::int                    AS llm_input_token_count,
    (ne."stats"->>'output_token_count')::int                   AS llm_output_token_count
FROM platform."CreditTransaction" c
LEFT JOIN platform."AgentNodeExecution" ne
       ON (c.metadata->>'node_exec_id') = ne."id"::text
WHERE c."createdAt" > CURRENT_DATE - INTERVAL '90 days'
