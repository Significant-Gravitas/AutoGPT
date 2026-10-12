-- =============================================================
-- View: analytics.graph_execution
-- Looker source alias: ds16  |  Charts: 21
-- =============================================================
-- DESCRIPTION
--   One row per agent graph execution (last 90 days).
--   Unpacks the JSONB stats column into individual numeric columns
--   and normalises the executionStatus — runs that failed due to
--   insufficient credits are reclassified as 'NO_CREDITS' for
--   easier filtering.  Error messages are scrubbed of IDs and URLs
--   to allow safe grouping.
--
-- SOURCE TABLES
--   platform.AgentGraphExecution  — Execution records
--   platform.AgentGraph           — Agent graph metadata (for name)
--   platform.LibraryAgent         — To flag possibly-AI (safe-mode) agents
--
-- OUTPUT COLUMNS
--   id                TEXT         Execution UUID
--   agentGraphId      TEXT         Agent graph UUID
--   agentGraphVersion INT          Graph version number
--   executionStatus   TEXT         COMPLETED | FAILED | NO_CREDITS | RUNNING | QUEUED | TERMINATED
--   createdAt         TIMESTAMPTZ  When the execution was queued
--   updatedAt         TIMESTAMPTZ  Last status update time
--   userId            TEXT         Owner user UUID
--   agentGraphName    TEXT         Human-readable agent name
--   cputime           DECIMAL      Total CPU seconds consumed
--   walltime          DECIMAL      Total wall-clock seconds
--   node_count        DECIMAL      Number of nodes in the graph
--   nodes_cputime     DECIMAL      CPU time across all nodes
--   nodes_walltime    DECIMAL      Wall time across all nodes
--   execution_cost    DECIMAL      Credit cost of this execution
--   correctness_score FLOAT        AI correctness score (if available)
--   possibly_ai       BOOLEAN      True if agent has sensitive_action_safe_mode enabled
--   groupedErrorMessage TEXT       Scrubbed error string (IDs/URLs replaced with wildcards)
--   expertId          TEXT         Hired expert the run is attributed to (nullable)
--   triggerSource     TEXT         How the run started: manual | api | schedule | webhook |
--                                  copilot | subgraph | admin. NULL for rows created
--                                  before the column existed (Sept 2026).
--   triggerRef        TEXT         What started it: schedule id, webhook id, chat session id,
--                                  'api_key'/'oauth', or the UI surface ('library', 'builder')
--   failureReason     TEXT         Structured failure reason (e.g. insufficient_balance)
--   isDryRun          BOOLEAN      LLM-simulated run; exclude from task counts
--   isSubgraphRun     BOOLEAN      Nested run started by a parent run; exclude from task counts
--   endedAt           TIMESTAMPTZ  When the run reached a terminal state (nullable)
--
-- WINDOW
--   Rolling 90 days (createdAt > CURRENT_DATE - 90 days)
--
-- EXAMPLE QUERIES
--   -- Daily execution counts by status
--   SELECT DATE_TRUNC('day', "createdAt") AS day, "executionStatus", COUNT(*)
--   FROM analytics.graph_execution
--   GROUP BY 1, 2 ORDER BY 1;
--
--   -- Average cost per execution by agent
--   SELECT "agentGraphName", AVG("execution_cost") AS avg_cost, COUNT(*) AS runs
--   FROM analytics.graph_execution
--   WHERE "executionStatus" = 'COMPLETED'
--   GROUP BY 1 ORDER BY avg_cost DESC;
--
--   -- Top error messages
--   SELECT "groupedErrorMessage", COUNT(*) AS occurrences
--   FROM analytics.graph_execution
--   WHERE "executionStatus" = 'FAILED'
--   GROUP BY 1 ORDER BY 2 DESC LIMIT 20;
-- =============================================================

SELECT
    ge."id"                                                        AS id,
    ge."agentGraphId"                                              AS agentGraphId,
    ge."agentGraphVersion"                                         AS agentGraphVersion,
    CASE
        -- Keep these legacy message predicates synchronized with
        -- backend/backend/util/exceptions.py and _LEGACY_BALANCE_CORPUS in its
        -- test module. New rows use failure_reason.
        WHEN CAST(ge."executionStatus" AS TEXT) = 'FAILED'
         AND (
                ge."stats"::jsonb->>'failure_reason' = 'insufficient_balance'
             OR (
                    ge."stats"::jsonb->>'failure_reason' IS NULL
                AND (
                       ge."stats"::jsonb->>'error' = 'You have no credits left to run an agent.'
                    OR ge."stats"::jsonb->>'error' ~ '^Insufficient balance of \$-?[0-9]+(\.[0-9]+)?, where this will cost \$-?[0-9]+(\.[0-9]+)?$'
                    OR ge."stats"::jsonb->>'error' ~ '^Insufficient balance to run [A-Za-z_][A-Za-z0-9_]*: dynamic-cost blocks require a positive balance\.$'
                    OR ge."stats"::jsonb->>'error' ~ '^Organization has -?[0-9]+ credits but needs [0-9]+$'
                   )
                )
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
    )                                                              AS groupedErrorMessage,
    -- Columns below were appended later; CREATE OR REPLACE VIEW only allows
    -- adding columns at the end, so keep new ones here. They are quoted so
    -- the column names stay camelCase as documented (an unquoted alias folds
    -- to lowercase); groupedErrorMessage above predates this and cannot be
    -- renamed in place.
    ge."expertId"                                                  AS "expertId",
    ge."triggerSource"                                             AS "triggerSource",
    ge."triggerRef"                                                AS "triggerRef",
    ge."stats"::jsonb->>'failure_reason'                           AS "failureReason",
    COALESCE((ge."stats"::jsonb->>'is_dry_run')::boolean, FALSE)   AS "isDryRun",
    ge."parentGraphExecutionId" IS NOT NULL                        AS "isSubgraphRun",
    ge."endedAt"                                                   AS "endedAt"
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
    ORDER BY "userId", "agentGraphId", "agentGraphVersion" DESC
) la ON la."userId" = ge."userId" AND la."agentGraphId" = ge."agentGraphId"
WHERE ge."createdAt" > CURRENT_DATE - INTERVAL '90 days'
