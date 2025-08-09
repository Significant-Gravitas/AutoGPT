-- Drop the existing view
DROP VIEW IF EXISTS "StoreSubmission";

-- Recreate the view with the new fields
CREATE VIEW "StoreSubmission" AS
SELECT
    sl.id AS listing_id,
    sl."owningUserId" AS user_id,
    slv."agentGraphId" AS agent_id,
    slv.version AS agent_version,
    sl.slug,
    COALESCE(slv.name, '') AS name,
    slv."subHeading" AS sub_heading,
    slv.description,
    slv."imageUrls" AS image_urls,
    slv."submittedAt" AS date_submitted,
    slv."submissionStatus" AS status,
    COALESCE(ar.run_count, 0::bigint) AS runs,
    COALESCE(avg(sr.score::numeric), 0.0)::double precision AS rating,
    slv.id AS store_listing_version_id,
    slv."reviewerId" AS reviewer_id,
    slv."reviewComments" AS review_comments,
    slv."internalComments" AS internal_comments,
    slv."reviewedAt" AS reviewed_at,
    slv."changesSummary" AS changes_summary,
    -- Add the two new fields:
    slv."videoUrl" AS video_url,
    slv.categories
FROM "StoreListing" sl
    JOIN "StoreListingVersion" slv ON slv."storeListingId" = sl.id
    LEFT JOIN "StoreListingReview" sr ON sr."storeListingVersionId" = slv.id
    LEFT JOIN (
        SELECT "AgentGraphExecution"."agentGraphId", count(*) AS run_count
        FROM "AgentGraphExecution"
        GROUP BY "AgentGraphExecution"."agentGraphId"
    ) ar ON ar."agentGraphId" = slv."agentGraphId"
WHERE sl."isDeleted" = false
GROUP BY sl.id, sl."owningUserId", slv.id, slv."agentGraphId", slv.version, sl.slug, slv.name,
         slv."subHeading", slv.description, slv."imageUrls", slv."submittedAt",
         slv."submissionStatus", slv."reviewerId", slv."reviewComments", slv."internalComments",
         slv."reviewedAt", slv."changesSummary", slv."videoUrl", slv.categories, ar.run_count; 