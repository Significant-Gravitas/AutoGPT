-- =============================================================
-- View: analytics.auth_activities
-- Looker source alias: ds49  |  Charts: 1
-- =============================================================
-- DESCRIPTION
--   Tracks authentication events (login, logout, SSO, password
--   reset, etc.) from Supabase's internal audit log.
--   Useful for monitoring sign-in patterns and detecting anomalies.
--
-- SOURCE TABLES
--   auth.audit_log_entries  — Supabase internal auth event log
--
-- OUTPUT COLUMNS
--   created_at      TIMESTAMPTZ  When the auth event occurred
--   actor_id        TEXT         User ID who triggered the event
--   actor_via_sso   TEXT         Whether the action was via SSO ('true'/'false')
--   action          TEXT         Event type (e.g. 'login', 'logout', 'token_refreshed')
--
-- WINDOW
--   Rolling 90 days from current date
--
-- EXAMPLE QUERIES
--   -- Daily login counts
--   SELECT DATE_TRUNC('day', created_at) AS day, COUNT(*) AS logins
--   FROM analytics.auth_activities
--   WHERE action = 'login'
--   GROUP BY 1 ORDER BY 1;
--
--   -- SSO vs password login breakdown
--   SELECT actor_via_sso, COUNT(*) FROM analytics.auth_activities
--   WHERE action = 'login' GROUP BY 1;
-- =============================================================

SELECT
    created_at,
    payload->>'actor_id'      AS actor_id,
    payload->>'actor_via_sso' AS actor_via_sso,
    payload->>'action'        AS action
FROM auth.audit_log_entries
WHERE created_at >= NOW() - INTERVAL '90 days'
