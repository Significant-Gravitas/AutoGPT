-- =============================================================
-- View: analytics.chat_turn
-- Looker source alias: (new)  |  Charts: 0
-- =============================================================
-- DESCRIPTION
--   One row per user-role chat message (last 90 days): the copilot
--   equivalent of a graph execution. This is where "run_autopilot"
--   and "run_expert" come from for Looker, and it is the SQL twin of
--   the PostHog events of the same names.
--
--   Dream/daydream sessions (memory maintenance, not the user) are
--   excluded. Sessions with origin 'automation' are model-authored
--   (scheduled follow-ups fired into a fresh chat); their user-role
--   opener is kept but flagged so human activity can be isolated.
--
-- SOURCE TABLES
--   platform.ChatMessage  — messages (role, sequence, model)
--   platform.ChatSession  — owner, expert scope, origin/kind metadata
--
-- OUTPUT COLUMNS
--   id                     TEXT         Message UUID
--   createdAt              TIMESTAMPTZ  When the turn was sent
--   sessionId              TEXT         Chat session UUID
--   userId                 TEXT         Session owner
--   organizationId         TEXT         Org the session belongs to (nullable)
--   expertId               TEXT         Expert the session is scoped to (NULL = Autopilot)
--   surface                TEXT         'autopilot' | 'expert'
--   sessionOrigin          TEXT         'interactive' | 'automation' (legacy rows -> 'interactive')
--   isHumanTurn            BOOLEAN      sessionOrigin <> 'automation'
--   sourcePlatform         TEXT         'slack' | 'telegram' | 'discord' | NULL (= web chat)
--   sequence               INT          Position of the message in the session
--   isFirstTurnInSession   BOOLEAN      First user turn of the session (looks past the window)
--   sessionCreatedAt       TIMESTAMPTZ  When the session was created
--
-- WINDOW
--   Rolling 90 days (createdAt > CURRENT_DATE - 90 days)
--
-- EXAMPLE QUERIES
--   -- Daily human copilot turns by surface
--   SELECT DATE_TRUNC('day', "createdAt") AS day, surface, COUNT(*)
--   FROM analytics.chat_turn
--   WHERE "isHumanTurn"
--   GROUP BY 1, 2 ORDER BY 1;
--
--   -- Users who talked to an expert this week
--   SELECT COUNT(DISTINCT "userId") FROM analytics.chat_turn
--   WHERE surface = 'expert' AND "isHumanTurn"
--     AND "createdAt" >= DATE_TRUNC('week', CURRENT_DATE);
-- =============================================================

SELECT
    m."id"                                                        AS id,
    m."createdAt"                                                 AS "createdAt",
    m."sessionId"                                                 AS "sessionId",
    s."userId"                                                    AS "userId",
    s."organizationId"                                            AS "organizationId",
    s."expertId"                                                  AS "expertId",
    CASE WHEN s."expertId" IS NULL THEN 'autopilot' ELSE 'expert' END
                                                                  AS surface,
    COALESCE(s."metadata"::jsonb->>'origin', 'interactive')       AS "sessionOrigin",
    COALESCE(s."metadata"::jsonb->>'origin', 'interactive') <> 'automation'
                                                                  AS "isHumanTurn",
    s."metadata"::jsonb->>'source_platform'                       AS "sourcePlatform",
    m."sequence"                                                  AS sequence,
    NOT EXISTS (
      SELECT 1 FROM platform."ChatMessage" p
      WHERE p."sessionId" = m."sessionId"
        AND p."role" = 'user'
        AND p."sequence" < m."sequence"
    )                                                             AS "isFirstTurnInSession",
    s."createdAt"                                                 AS "sessionCreatedAt"
FROM platform."ChatMessage" m
JOIN platform."ChatSession" s ON s."id" = m."sessionId"
WHERE m."role" = 'user'
  AND m."createdAt" > CURRENT_DATE - INTERVAL '90 days'
  AND COALESCE(s."metadata"::jsonb->>'kind', 'normal') <> 'dream'
