-- ============================================================================
-- Copilot building-mode economics: do sessions have long non-building tails
-- (exit would save money) and how often does building RESUME after a lull
-- (exit would cost money)?  Read-only; run on a replica if available.
--
-- Assumptions / caveats:
--  * Tables: platform."ChatSession", platform."ChatMessage" (adjust schema
--    qualifier if prod differs).
--  * Pre-#13593 history often lacks assistant "toolCalls" (the mid-turn-flush
--    persistence bug), so detection below primarily classifies TOOL-RESULT
--    rows by their JSON content->>'type' (reliable), with toolCalls as a
--    bonus signal where it survived. The type is extracted with a regex on
--    the first 200 chars (never throws) because tool-output truncation left
--    rows that start with '{' but are not valid JSON — a ::jsonb cast fails.
--  * "createdAt" is batch-assigned within a turn's flush; cross-turn deltas
--    (user message arrivals) are individually timestamped and are what the
--    cold-gap analysis uses.
--  * Date window kept to 90 days; widen/narrow as needed.
-- ============================================================================

-- ---------------------------------------------------------------------------
-- Q0. DISCOVERY: which tool-result types exist and how common are they.
-- Run this first; adjust the build_types list in later queries if needed.
-- ---------------------------------------------------------------------------
SELECT
  substring(left(m.content, 200) from '"type"\s*:\s*"([a-zA-Z_]+)"') AS result_type,
  count(*)                                                            AS n
FROM platform."ChatMessage" m
WHERE m.role = 'tool'
  AND m."createdAt" > now() - interval '90 days'
GROUP BY 1
ORDER BY n DESC
LIMIT 50;

-- ---------------------------------------------------------------------------
-- Shared CTE fragment used by Q1-Q4 (paste above each query, or run in one
-- session with \set; duplicated inline below for copy-paste convenience).
--
-- "build event" := a tool-result row whose type indicates agent-building
-- activity, OR an assistant row whose surviving toolCalls mention a build
-- tool. Adjust lists after inspecting Q0.
-- ---------------------------------------------------------------------------

-- ---------------------------------------------------------------------------
-- Q1. How many sessions are "building sessions" at all, and their sizes.
-- ---------------------------------------------------------------------------
WITH build_events AS (
  SELECT m."sessionId", m.sequence, m."createdAt"
  FROM platform."ChatMessage" m
  WHERE m."createdAt" > now() - interval '90 days'
    AND (
      (m.role = 'tool' AND substring(left(m.content, 200) from '"type"\s*:\s*"([a-zA-Z_]+)"') IN (
        'agent_builder_guide', 'agent_builder_validation_result',
        'agent_builder_fix_result', 'agent_saved', 'agent_preview'
      ))
      OR (m.role = 'assistant' AND m."toolCalls" IS NOT NULL AND EXISTS (
        SELECT 1 FROM jsonb_array_elements(m."toolCalls") tc
        WHERE tc -> 'function' ->> 'name' IN (
          'get_agent_building_guide', 'enter_agent_building_mode',
          'create_agent', 'edit_agent', 'validate_agent_graph',
          'fix_agent_graph', 'customize_agent'
        )
      ))
    )
),
sessions AS (
  SELECT s.id, count(m.id) AS total_msgs,
         count(*) FILTER (WHERE m.role = 'user') AS user_turns
  FROM platform."ChatSession" s
  JOIN platform."ChatMessage" m ON m."sessionId" = s.id
  WHERE s."updatedAt" > now() - interval '90 days'
  GROUP BY s.id
)
SELECT
  count(*)                                            AS sessions_total,
  count(*) FILTER (WHERE b."sessionId" IS NOT NULL)   AS building_sessions,
  round(100.0 * count(*) FILTER (WHERE b."sessionId" IS NOT NULL)
        / greatest(count(*), 1), 1)                   AS pct_building,
  percentile_cont(0.5) WITHIN GROUP (ORDER BY s.total_msgs)
    FILTER (WHERE b."sessionId" IS NOT NULL)          AS median_msgs_building
FROM sessions s
LEFT JOIN (SELECT DISTINCT "sessionId" FROM build_events) b
  ON b."sessionId" = s.id;

-- ---------------------------------------------------------------------------
-- Q2. THE SAVINGS POOL: for each building session, how much conversation
-- happens AFTER the last build event ("post-build tail")?  Exit only pays
-- off when these tails are long.
-- ---------------------------------------------------------------------------
WITH build_events AS (
  SELECT m."sessionId", m.sequence
  FROM platform."ChatMessage" m
  WHERE m."createdAt" > now() - interval '90 days'
    AND (
      (m.role = 'tool' AND substring(left(m.content, 200) from '"type"\s*:\s*"([a-zA-Z_]+)"') IN (
        'agent_builder_guide', 'agent_builder_validation_result',
        'agent_builder_fix_result', 'agent_saved', 'agent_preview'
      ))
      OR (m.role = 'assistant' AND m."toolCalls" IS NOT NULL AND EXISTS (
        SELECT 1 FROM jsonb_array_elements(m."toolCalls") tc
        WHERE tc -> 'function' ->> 'name' IN (
          'get_agent_building_guide', 'enter_agent_building_mode',
          'create_agent', 'edit_agent', 'validate_agent_graph',
          'fix_agent_graph', 'customize_agent'
        )
      ))
    )
),
last_build AS (
  SELECT "sessionId", max(sequence) AS last_build_seq
  FROM build_events GROUP BY "sessionId"
),
tails AS (
  SELECT lb."sessionId",
         count(*) FILTER (WHERE m.sequence > lb.last_build_seq)      AS tail_msgs,
         count(*) FILTER (WHERE m.sequence > lb.last_build_seq
                            AND m.role = 'user')                     AS tail_user_turns
  FROM last_build lb
  JOIN platform."ChatMessage" m ON m."sessionId" = lb."sessionId"
  GROUP BY lb."sessionId"
)
SELECT
  count(*)                                                   AS building_sessions,
  percentile_cont(0.5)  WITHIN GROUP (ORDER BY tail_user_turns) AS p50_tail_user_turns,
  percentile_cont(0.9)  WITHIN GROUP (ORDER BY tail_user_turns) AS p90_tail_user_turns,
  percentile_cont(0.99) WITHIN GROUP (ORDER BY tail_user_turns) AS p99_tail_user_turns,
  count(*) FILTER (WHERE tail_user_turns >= 10)              AS sessions_tail_ge_10_turns,
  count(*) FILTER (WHERE tail_user_turns >= 25)              AS sessions_tail_ge_25_turns
FROM tails;

-- ---------------------------------------------------------------------------
-- Q3. THE RE-ENTRY RISK: within building sessions, how often does building
-- RESUME after a lull (>= 1 intervening user turn AND a cold gap > 6 min)?
-- Each such resume is what a premature exit would turn into a double
-- cache-write (~$0.4 at 100K context).
-- ---------------------------------------------------------------------------
WITH build_events AS (
  SELECT m."sessionId", m.sequence, m."createdAt"
  FROM platform."ChatMessage" m
  WHERE m."createdAt" > now() - interval '90 days'
    AND (
      (m.role = 'tool' AND substring(left(m.content, 200) from '"type"\s*:\s*"([a-zA-Z_]+)"') IN (
        'agent_builder_guide', 'agent_builder_validation_result',
        'agent_builder_fix_result', 'agent_saved', 'agent_preview'
      ))
      OR (m.role = 'assistant' AND m."toolCalls" IS NOT NULL AND EXISTS (
        SELECT 1 FROM jsonb_array_elements(m."toolCalls") tc
        WHERE tc -> 'function' ->> 'name' IN (
          'get_agent_building_guide', 'enter_agent_building_mode',
          'create_agent', 'edit_agent', 'validate_agent_graph',
          'fix_agent_graph', 'customize_agent'
        )
      ))
    )
),
gaps AS (
  SELECT b."sessionId",
         b.sequence,
         lag(b.sequence)  OVER w AS prev_seq,
         lag(b."createdAt") OVER w AS prev_at,
         b."createdAt"
  FROM build_events b
  WINDOW w AS (PARTITION BY b."sessionId" ORDER BY b.sequence)
),
reentries AS (
  SELECT g.*,
         (SELECT count(*) FROM platform."ChatMessage" u
          WHERE u."sessionId" = g."sessionId" AND u.role = 'user'
            AND u.sequence > g.prev_seq AND u.sequence < g.sequence)
           AS intervening_user_turns
  FROM gaps g
  WHERE g.prev_seq IS NOT NULL
    AND g."createdAt" - g.prev_at > interval '6 minutes'
)
SELECT
  count(DISTINCT "sessionId") FILTER (WHERE intervening_user_turns >= 1)
    AS sessions_with_cold_reentry,
  (SELECT count(DISTINCT "sessionId") FROM build_events)
    AS building_sessions_total,
  round(100.0 * count(DISTINCT "sessionId")
          FILTER (WHERE intervening_user_turns >= 1)
        / greatest((SELECT count(DISTINCT "sessionId") FROM build_events), 1), 1)
    AS pct_sessions_cold_reentry,
  count(*) FILTER (WHERE intervening_user_turns >= 1)
    AS cold_reentry_events_total
FROM reentries;

-- ---------------------------------------------------------------------------
-- Q4. COLD STARTS per building session (context: every cold start already
-- pays a conversation cache-write; carrying the guide adds only ~9K to it).
-- A "cold start" = a user message arriving > 6 min after the previous
-- message in the session.
-- ---------------------------------------------------------------------------
WITH building AS (
  SELECT DISTINCT m."sessionId" AS id
  FROM platform."ChatMessage" m
  WHERE m."createdAt" > now() - interval '90 days'
    AND m.role = 'tool'
    AND substring(left(m.content, 200) from '"type"\s*:\s*"([a-zA-Z_]+)"') IN (
      'agent_builder_guide', 'agent_builder_validation_result',
      'agent_builder_fix_result', 'agent_saved', 'agent_preview'
    )
),
deltas AS (
  SELECT m."sessionId", m.sequence, m.role,
         m."createdAt" - lag(m."createdAt")
           OVER (PARTITION BY m."sessionId" ORDER BY m.sequence) AS gap
  FROM platform."ChatMessage" m
  JOIN building b ON b.id = m."sessionId"
),
per_session AS (
  SELECT "sessionId",
         count(*) FILTER (WHERE role = 'user' AND gap > interval '6 minutes')
           AS cold_starts
  FROM deltas GROUP BY "sessionId"
)
SELECT
  percentile_cont(0.5) WITHIN GROUP (ORDER BY cold_starts) AS p50_cold_starts,
  percentile_cont(0.9) WITHIN GROUP (ORDER BY cold_starts) AS p90_cold_starts,
  avg(cold_starts)                                         AS avg_cold_starts
FROM per_session;

-- ---------------------------------------------------------------------------
-- Interpretation guide:
--  * Exit SAVES ≈ tail_user_turns × requests_per_turn × 9K × cache-read rate
--    (+ 9K × cache-write per post-exit cold start).  With Sonnet 5 that is
--    roughly $0.003/request and $0.034/cold-start.
--  * Exit COSTS ≈ (per cold re-entry) one extra conversation cache-write,
--    ≈ context_tokens × $3.75/M ≈ $0.375 at 100K.
--  * Rule of thumb: exit is net-positive only if
--      p50_tail_user_turns is large  AND  pct_sessions_cold_reentry is small.
--    If Q3's pct is above ~10-20% and Q2's median tail is < ~15 user turns,
--    the exit feature loses money; prefer keeping building mode one-way.
-- ============================================================================
