-- =============================================================
-- View: analytics.user_onboarding_integration
-- Looker source alias: ds75  |  Charts: 1
-- =============================================================
-- DESCRIPTION
--   Pre-aggregated count of users who selected each integration
--   during onboarding.  One row per integration type, sorted
--   by popularity.
--
-- SOURCE TABLES
--   platform.UserOnboarding  — integrations array column
--
-- OUTPUT COLUMNS
--   integration            TEXT    Integration name (e.g. 'github', 'slack', 'notion')
--   users_with_integration BIGINT  Distinct users who selected this integration
--
-- WINDOW
--   Users who started onboarding in the last 90 days
--
-- EXAMPLE QUERIES
--   -- Full integration popularity ranking
--   SELECT * FROM analytics.user_onboarding_integration;
--
--   -- Top 5 integrations
--   SELECT * FROM analytics.user_onboarding_integration LIMIT 5;
-- =============================================================

WITH exploded AS (
  SELECT
      u."userId" AS user_id,
      UNNEST(u."integrations") AS integration
  FROM platform."UserOnboarding" u
  WHERE u."createdAt" >= CURRENT_DATE - INTERVAL '90 days'
)
SELECT
    integration,
    COUNT(DISTINCT user_id) AS users_with_integration
FROM exploded
WHERE integration IS NOT NULL AND integration <> ''
GROUP BY integration
ORDER BY users_with_integration DESC
