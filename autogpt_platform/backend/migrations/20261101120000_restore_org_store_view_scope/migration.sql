-- 20261020130000_fix_agent_run_counts_graph_version rebuilt the store views
-- from their definitions before 20260827150000_store_submission_exact_scope
-- and 20260827170000_store_creator_live_visibility, and it sorts after both,
-- so it dropped StoreSubmission's exact version scope (organization_id and
-- team_id from the version) and StoreCreator's live-version rule. Restore
-- both on top of its run-count fix, keeping StoreCreator's per-listing
-- aggregation from #15762 so reviews and categories cannot multiply runs.

CREATE OR REPLACE VIEW "StoreSubmission" AS
WITH review_stats AS (
    SELECT
        "storeListingVersionId" AS version_id,
        avg(score)              AS avg_rating,
        count(*)                AS review_count
    FROM "StoreListingReview"
    GROUP BY "storeListingVersionId"
)
SELECT
    sl.id              AS listing_id,
    sl."owningUserId"  AS user_id,
    sl.slug            AS slug,

    slv.id                   AS listing_version_id,
    slv.version              AS listing_version,
    slv."agentGraphId"       AS graph_id,
    slv."agentGraphVersion"  AS graph_version,
    slv.name                 AS name,
    slv."subHeading"         AS sub_heading,
    slv.description          AS description,
    slv.instructions         AS instructions,
    slv.categories           AS categories,
    slv."imageUrls"          AS image_urls,
    slv."videoUrl"           AS video_url,
    slv."agentOutputDemoUrl" AS agent_output_demo_url,
    slv."submittedAt"        AS submitted_at,
    slv."changesSummary"     AS changes_summary,
    slv."submissionStatus"   AS status,
    slv."reviewedAt"         AS reviewed_at,
    slv."reviewerId"         AS reviewer_id,
    slv."reviewComments"     AS review_comments,
    slv."internalComments"   AS internal_comments,
    slv."isDeleted"          AS is_deleted,

    COALESCE(run_stats.run_count, 0::bigint) AS run_count,
    COALESCE(review_stats.review_count, 0::bigint) AS review_count,
    COALESCE(review_stats.avg_rating, 0.0)::double precision AS review_avg_rating,

    slv."organizationId"     AS organization_id,
    slv."teamId"             AS team_id
FROM      "StoreListing"        sl
JOIN      "StoreListingVersion" slv     ON slv."storeListingId" = sl.id
LEFT JOIN review_stats                  ON review_stats.version_id = slv.id
LEFT JOIN mv_agent_run_counts run_stats ON run_stats.graph_id = slv."agentGraphId"
WHERE     sl."isDeleted" = false;


CREATE OR REPLACE VIEW "StoreCreator" AS
WITH creator_listings AS (
    SELECT
        sl.id AS listing_id,
        sl."owningUserId",
        slv.id AS version_id,
        slv."agentGraphId",
        slv.categories
    FROM "StoreListing" sl
    JOIN "StoreListingVersion" slv
      ON slv.id = sl."activeVersionId"
     AND slv."storeListingId" = sl.id
     AND slv."submissionStatus" = 'APPROVED'
     AND slv."isAvailable" = true
     AND slv."isDeleted" = false
    WHERE sl."isDeleted" = false
      AND sl."hasApprovedVersion" = true
),
creator_runs AS (
    SELECT
        cl."owningUserId",
        COUNT(*)                        AS num_agents,
        SUM(COALESCE(ar.run_count, 0))  AS agent_runs
    FROM creator_listings cl
    LEFT JOIN "mv_agent_run_counts" ar
           ON ar.graph_id = cl."agentGraphId"
    GROUP BY cl."owningUserId"
),
creator_ratings AS (
    SELECT
        cl."owningUserId",
        AVG(sr.score::numeric) AS agent_rating
    FROM creator_listings cl
    JOIN "StoreListingReview" sr
      ON sr."storeListingVersionId" = cl.version_id
    GROUP BY cl."owningUserId"
),
creator_categories AS (
    SELECT
        cl."owningUserId",
        array_agg(DISTINCT cat ORDER BY cat)
          FILTER (WHERE cat IS NOT NULL AND cat != '') AS all_categories
    FROM creator_listings cl
    LEFT JOIN LATERAL unnest(COALESCE(cl.categories, ARRAY[]::text[])) AS cat ON true
    GROUP BY cl."owningUserId"
)
SELECT
    p.username,
    p.name,
    p."avatarUrl"                                AS avatar_url,
    p.description,
    COALESCE(cc.all_categories, ARRAY[]::text[]) AS top_categories,
    p.links,
    p."isFeatured"                               AS is_featured,
    COALESCE(cr.num_agents, 0::bigint)           AS num_agents,
    COALESCE(crt.agent_rating, 0.0)              AS agent_rating,
    COALESCE(cr.agent_runs, 0::numeric)          AS agent_runs,
    o.name                                       AS org_name
FROM "Profile" p
LEFT JOIN creator_runs cr        ON cr."owningUserId" = p."userId"
LEFT JOIN creator_ratings crt    ON crt."owningUserId" = p."userId"
LEFT JOIN creator_categories cc  ON cc."owningUserId" = p."userId"
LEFT JOIN "OrgMember" om ON om."userId" = p."userId" AND om."isOwner" = true
LEFT JOIN "Organization" o ON o.id = om."orgId" AND o."isPersonal" = true;
