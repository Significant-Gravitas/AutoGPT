-- =============================================================
-- View: analytics.user_onboarding
-- Looker source alias: ds68  |  Charts: 3
-- =============================================================
-- DESCRIPTION
--   One row per user onboarding record.  Contains the user's
--   stated usage reason, selected integrations, completed
--   onboarding steps and optional first agent selection.
--   Full history (no date filter) since onboarding happens
--   once per user.
--
-- SOURCE TABLES
--   platform.UserOnboarding  — Onboarding state per user
--
-- OUTPUT COLUMNS
--   id                            TEXT         Onboarding record UUID
--   createdAt                     TIMESTAMPTZ  When onboarding started
--   updatedAt                     TIMESTAMPTZ  Last update to onboarding state
--   usageReason                   TEXT         Why user signed up (e.g. 'work', 'personal')
--   integrations                  TEXT[]       Array of integration names the user selected
--   userId                        TEXT         User UUID
--   completedSteps                TEXT[]       Array of onboarding step enums completed
--   selectedStoreListingVersionId TEXT         First marketplace agent the user chose (if any)
--
-- EXAMPLE QUERIES
--   -- Usage reason breakdown
--   SELECT "usageReason", COUNT(*) FROM analytics.user_onboarding GROUP BY 1;
--
--   -- Completion rate per step
--   SELECT step, COUNT(*) AS users_completed
--   FROM analytics.user_onboarding
--   CROSS JOIN LATERAL UNNEST("completedSteps") AS step
--   GROUP BY 1 ORDER BY users_completed DESC;
-- =============================================================

SELECT
    id,
    "createdAt",
    "updatedAt",
    "usageReason",
    integrations,
    "userId",
    "completedSteps",
    "selectedStoreListingVersionId"
FROM platform."UserOnboarding"
