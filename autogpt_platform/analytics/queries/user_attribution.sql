-- =============================================================
-- View: analytics.user_attribution
-- Looker source alias: (new)  |  Charts: 0
-- =============================================================
-- DESCRIPTION
--   One row per user: where they came from, captured once around
--   signup by the browser that created the account. Carries the ids
--   the analytics tools knew the user by before they had a user id,
--   so channel (DataFast), product analytics (PostHog) and flag
--   bucketing (LaunchDarkly, via the shared anonymous id) can all be
--   joined to activation, retention and revenue on user_id.
--
--   Join to analytics.user_lifecycle for "which channel brings users
--   who activate". Rows exist only for users who signed up (or first
--   logged in) after this table shipped.
--
-- SOURCE TABLES
--   platform.UserAttribution
--
-- OUTPUT COLUMNS
--   user_id              TEXT         User UUID
--   captured_at          TIMESTAMPTZ  When the row was first written
--   anonymous_id         TEXT         First-party anonymous id shared with PostHog + LaunchDarkly
--   posthog_distinct_id  TEXT         PostHog device id at signup, when different
--   datafast_visitor_id  TEXT         DataFast visitor (join key into DataFast exports)
--   datafast_session_id  TEXT         DataFast session that contained the signup
--   landing_path         TEXT         First page seen in that browser
--   referrer             TEXT         document.referrer on that first page
--   utm_source / utm_medium / utm_campaign  TEXT   From the first landing URL
--   signup_method        TEXT         'email' | 'google' | NULL (not a fresh signup in that browser)
--
-- EXAMPLE QUERIES
--   -- Activation rate by UTM source
--   SELECT COALESCE(a.utm_source, '(none)') AS source,
--          COUNT(*) AS signups,
--          AVG(CASE WHEN l.activated THEN 1 ELSE 0 END) AS activation_rate
--   FROM analytics.user_attribution a
--   JOIN analytics.user_lifecycle l ON l.user_id = a.user_id
--   WHERE l.signup_at < NOW() - INTERVAL '14 days'
--   GROUP BY 1 ORDER BY signups DESC;
-- =============================================================

SELECT
    "userId"            AS user_id,
    "createdAt"         AS captured_at,
    "anonymousId"       AS anonymous_id,
    "posthogDistinctId" AS posthog_distinct_id,
    "datafastVisitorId" AS datafast_visitor_id,
    "datafastSessionId" AS datafast_session_id,
    "landingPath"       AS landing_path,
    "referrer"          AS referrer,
    "utmSource"         AS utm_source,
    "utmMedium"         AS utm_medium,
    "utmCampaign"       AS utm_campaign,
    "signupMethod"      AS signup_method
FROM platform."UserAttribution"
