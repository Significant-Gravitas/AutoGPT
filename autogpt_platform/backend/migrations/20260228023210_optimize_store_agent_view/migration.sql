BEGIN;

DROP VIEW IF EXISTS "StoreAgent";

-- Recreate the StoreAgent view with the following changes:
-- 1. Add `recommendedScheduleCron` column from `StoreListingVersion`
-- 2. Narrow to *explicitly active* version rather than *highest* version
CREATE OR REPLACE VIEW "StoreAgent" AS
WITH store_agent_versions AS (
    SELECT
        "storeListingId",
        array_agg(DISTINCT version::text ORDER BY version::text) AS versions
    FROM "StoreListingVersion"
    WHERE "submissionStatus" = 'APPROVED'
    GROUP BY "storeListingId"
),
agent_graph_versions AS (
    SELECT
        "storeListingId",
        array_agg(DISTINCT "agentGraphVersion"::text ORDER BY "agentGraphVersion"::text) AS graph_versions
    FROM "StoreListingVersion"
    WHERE "submissionStatus" = 'APPROVED'
    GROUP BY "storeListingId"
)
SELECT
     sl.id                                                       AS listing_id,
     slv.id                                                      AS listing_version_id,
     slv."createdAt"                                             AS updated_at,
     sl.slug,
     COALESCE(slv.name, '')                                      AS agent_name,
     slv."videoUrl"                                              AS agent_video,
     slv."agentOutputDemoUrl"                                    AS agent_output_demo,
     COALESCE(slv."imageUrls", ARRAY[]::text[])                  AS agent_image,
     slv."isFeatured"                                            AS featured,
     cp.username                                                 AS creator_username,
     cp."avatarUrl"                                              AS creator_avatar,
     slv."subHeading"                                            AS sub_heading,
     slv.description,
     slv.categories,
     COALESCE(arc.run_count, 0::bigint)                          AS runs,
     COALESCE(reviews.avg_rating, 0.0)::double precision         AS rating,
     COALESCE(sav.versions, ARRAY[slv.version::text])            AS versions,
     slv."agentGraphId"                                          AS graph_id,
     COALESCE(
        agv.graph_versions, ARRAY[slv."agentGraphVersion"::text]
     )                                                           AS graph_versions,
     slv."isAvailable"                                           AS is_available,
     COALESCE(sl."useForOnboarding", false)                      AS use_for_onboarding,
     slv."recommendedScheduleCron"                               AS recommended_schedule_cron
FROM "StoreListing"             AS sl
JOIN "StoreListingVersion"      AS slv
  ON slv."storeListingId" = sl.id
 AND slv.id = sl."activeVersionId"
 AND slv."submissionStatus" = 'APPROVED'
JOIN "AgentGraph"               AS ag
  ON slv."agentGraphId" = ag.id
 AND slv."agentGraphVersion" = ag.version
LEFT JOIN "Profile"             AS cp
  ON sl."owningUserId" = cp."userId"
LEFT JOIN "mv_review_stats"     AS reviews
  ON sl.id = reviews."storeListingId"
LEFT JOIN "mv_agent_run_counts" AS arc
  ON ag.id = arc."agentGraphId"
LEFT JOIN store_agent_versions  AS sav
  ON sl.id = sav."storeListingId"
LEFT JOIN agent_graph_versions  AS agv
  ON sl.id = agv."storeListingId"
WHERE sl."isDeleted" = false
  AND sl."hasApprovedVersion" = true;

COMMIT;
