-- =============================================================
-- View: analytics.node_block_execution
-- Looker source alias: ds14  |  Charts: 11
-- =============================================================
-- DESCRIPTION
--   One row per node (block) execution (last 90 days).
--   Unpacks stats JSONB and joins to identify which block type
--   was run.  For failed nodes, joins the error output and
--   scrubs it for safe grouping.
--
-- SOURCE TABLES
--   platform.AgentNodeExecution              — Node execution records
--   platform.AgentNode                       — Node → block mapping
--   platform.AgentBlock                      — Block name/ID
--   platform.AgentNodeExecutionInputOutput   — Error output values
--
-- OUTPUT COLUMNS
--   id                    TEXT         Node execution UUID
--   agentGraphExecutionId TEXT         Parent graph execution UUID
--   agentNodeId           TEXT         Node UUID within the graph
--   executionStatus       TEXT         COMPLETED | FAILED | QUEUED | RUNNING | TERMINATED
--   addedTime             TIMESTAMPTZ  When the node was queued
--   queuedTime            TIMESTAMPTZ  When it entered the queue
--   startedTime           TIMESTAMPTZ  When execution started
--   endedTime             TIMESTAMPTZ  When execution finished
--   inputSize             BIGINT       Input payload size in bytes
--   outputSize            BIGINT       Output payload size in bytes
--   walltime              NUMERIC      Wall-clock seconds for this node
--   cputime               NUMERIC      CPU seconds for this node
--   llmRetryCount         INT          Number of LLM retries
--   llmCallCount          INT          Number of LLM API calls made
--   inputTokenCount       BIGINT       LLM input tokens consumed
--   outputTokenCount      BIGINT       LLM output tokens produced
--   blockName             TEXT         Human-readable block name (e.g. 'OpenAIBlock')
--   blockId               TEXT         Block UUID
--   groupedErrorMessage   TEXT         Scrubbed error (IDs/URLs wildcarded)
--   errorMessage          TEXT         Raw error output (only set when FAILED)
--
-- WINDOW
--   Rolling 90 days (addedTime > CURRENT_DATE - 90 days)
--
-- EXAMPLE QUERIES
--   -- Most-used blocks by execution count
--   SELECT "blockName", COUNT(*) AS executions,
--          COUNT(*) FILTER (WHERE "executionStatus"='FAILED') AS failures
--   FROM analytics.node_block_execution
--   GROUP BY 1 ORDER BY executions DESC LIMIT 20;
--
--   -- Average LLM token usage per block
--   SELECT "blockName",
--          AVG("inputTokenCount") AS avg_input_tokens,
--          AVG("outputTokenCount") AS avg_output_tokens
--   FROM analytics.node_block_execution
--   WHERE "llmCallCount" > 0
--   GROUP BY 1 ORDER BY avg_input_tokens DESC;
--
--   -- Top failure reasons
--   SELECT "blockName", "groupedErrorMessage", COUNT(*) AS count
--   FROM analytics.node_block_execution
--   WHERE "executionStatus" = 'FAILED'
--   GROUP BY 1, 2 ORDER BY count DESC LIMIT 20;
-- =============================================================

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
WHERE ne."addedTime" > CURRENT_DATE - INTERVAL '90 days'
