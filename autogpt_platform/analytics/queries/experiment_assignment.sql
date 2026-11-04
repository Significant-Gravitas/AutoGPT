-- =============================================================
-- View: analytics.experiment_assignment
-- Looker source alias: (new)  |  Charts: 0
-- =============================================================
-- DESCRIPTION
--   One row per (user, experiment): the A/B/C arm the user was
--   bucketed into the first time the experiment was evaluated for
--   them. Join on user_id to split any other view (retention, tasks,
--   unit economics) by arm. PostHog owns significance testing; this
--   is the durable copy so Looker can segment by variant too.
--
-- SOURCE TABLES
--   platform.ExperimentAssignment — first-seen arm per user/experiment
--
-- OUTPUT COLUMNS
--   user_id          TEXT         User UUID
--   experiment_key   TEXT         PostHog flag key (e.g. 'subscription-pricing-page-initial-state')
--   variant          TEXT         Arm label ('control', 'yearly-pro', ...)
--   source           TEXT         'posthog' | 'backend'
--   assigned_at      TIMESTAMPTZ  First time the arm was observed for this user
--
-- WINDOW
--   Full history (assignments are small and never rewritten)
--
-- EXAMPLE QUERIES
--   -- Arm sizes per experiment
--   SELECT experiment_key, variant, COUNT(*) AS users
--   FROM analytics.experiment_assignment
--   GROUP BY 1, 2 ORDER BY 1, 2;
--
--   -- Activation rate by arm
--   SELECT a.variant,
--          AVG(CASE WHEN l.activated THEN 1 ELSE 0 END) AS activation_rate
--   FROM analytics.experiment_assignment a
--   JOIN analytics.user_lifecycle l ON l.user_id = a.user_id
--   WHERE a.experiment_key = 'subscription-pricing-page-initial-state'
--   GROUP BY 1;
-- =============================================================

SELECT
    "userId"        AS user_id,
    "experimentKey" AS experiment_key,
    "variant"       AS variant,
    "source"        AS source,
    "createdAt"     AS assigned_at
FROM platform."ExperimentAssignment"
