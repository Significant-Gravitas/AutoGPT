CREATE OR REPLACE VIEW "StoreSubmission" AS
WITH review_stats AS (
    SELECT
        "storeListingVersionId" AS version_id,
        avg(score) AS avg_rating,
        count(*) AS review_count
    FROM "StoreListingReview"
    GROUP BY "storeListingVersionId"
)
SELECT
    sl.id AS listing_id,
    sl."owningUserId" AS user_id,
    sl.slug AS slug,
    slv.id AS listing_version_id,
    slv.version AS listing_version,
    slv."agentGraphId" AS graph_id,
    slv."agentGraphVersion" AS graph_version,
    slv.name AS name,
    slv."subHeading" AS sub_heading,
    slv.description AS description,
    slv.instructions AS instructions,
    slv.categories AS categories,
    slv."imageUrls" AS image_urls,
    slv."videoUrl" AS video_url,
    slv."agentOutputDemoUrl" AS agent_output_demo_url,
    slv."submittedAt" AS submitted_at,
    slv."changesSummary" AS changes_summary,
    slv."submissionStatus" AS status,
    slv."reviewedAt" AS reviewed_at,
    slv."reviewerId" AS reviewer_id,
    slv."reviewComments" AS review_comments,
    slv."internalComments" AS internal_comments,
    slv."isDeleted" AS is_deleted,
    COALESCE(run_stats.run_count, 0::bigint) AS run_count,
    COALESCE(review_stats.review_count, 0::bigint) AS review_count,
    COALESCE(review_stats.avg_rating, 0.0)::double precision AS review_avg_rating,
    slv."organizationId" AS organization_id,
    slv."teamId" AS team_id
FROM "StoreListing" sl
JOIN "StoreListingVersion" slv ON slv."storeListingId" = sl.id
LEFT JOIN review_stats ON review_stats.version_id = slv.id
LEFT JOIN mv_agent_run_counts run_stats
    ON run_stats.graph_id = slv."agentGraphId"
WHERE sl."isDeleted" = false;
