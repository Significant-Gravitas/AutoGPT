-- AlterTable
ALTER TABLE "StoreListingVersion" ADD COLUMN "search" tsvector DEFAULT ''::tsvector;

-- Add trigger to update the search column with the tsvector of the agent
-- Function to be invoked by trigger

-- Drop the trigger first
DROP TRIGGER IF EXISTS "update_tsvector" ON "StoreListingVersion";

-- Drop the function completely
DROP FUNCTION IF EXISTS update_tsvector_column();

-- Now recreate it fresh
CREATE OR REPLACE FUNCTION update_tsvector_column() RETURNS TRIGGER AS $$
BEGIN
  NEW.search := to_tsvector('english', 
    COALESCE(NEW.name, '') || ' ' ||
    COALESCE(NEW.description, '') || ' ' ||
    COALESCE(NEW."subHeading", '')
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path = platform, pg_temp;

-- Recreate the trigger
CREATE TRIGGER "update_tsvector"
BEFORE INSERT OR UPDATE ON "StoreListingVersion"
FOR EACH ROW
EXECUTE FUNCTION update_tsvector_column();

UPDATE "StoreListingVersion"
SET search = to_tsvector('english', 
    COALESCE(name, '') || ' ' ||
    COALESCE(description, '') || ' ' ||
    COALESCE("subHeading", '')
)
WHERE search IS NULL;

-- Drop and recreate the StoreAgent view with isAvailable field
DROP VIEW IF EXISTS "StoreAgent";

CREATE OR REPLACE VIEW "StoreAgent" AS
WITH latest_versions AS (
    SELECT 
        "storeListingId",
        MAX(version) AS max_version
    FROM "StoreListingVersion"
    WHERE "submissionStatus" = 'APPROVED'
    GROUP BY "storeListingId"
),
agent_versions AS (
    SELECT 
        "storeListingId",
        array_agg(DISTINCT version::text ORDER BY version::text) AS versions
    FROM "StoreListingVersion"
    WHERE "submissionStatus" = 'APPROVED'
    GROUP BY "storeListingId"
)
SELECT
    sl.id AS listing_id,
    slv.id AS "storeListingVersionId",
    slv."createdAt" AS updated_at,
    sl.slug,
    COALESCE(slv.name, '') AS agent_name,
    slv."videoUrl" AS agent_video,
    COALESCE(slv."imageUrls", ARRAY[]::text[]) AS agent_image,
    slv."isFeatured" AS featured,
    p.username AS creator_username,  -- Allow NULL for malformed sub-agents
    p."avatarUrl" AS creator_avatar,  -- Allow NULL for malformed sub-agents
    slv."subHeading" AS sub_heading,
    slv.description,
    slv.categories,
    slv.search,
    COALESCE(ar.run_count, 0::bigint) AS runs,
    COALESCE(rs.avg_rating, 0.0)::double precision AS rating,
    COALESCE(av.versions, ARRAY[slv.version::text]) AS versions,
    COALESCE(sl."useForOnboarding", false) AS "useForOnboarding",
    slv."isAvailable" AS is_available  -- Add isAvailable field to filter sub-agents
FROM "StoreListing" sl
JOIN latest_versions lv 
    ON sl.id = lv."storeListingId"
JOIN "StoreListingVersion" slv 
    ON slv."storeListingId" = lv."storeListingId"
   AND slv.version = lv.max_version
   AND slv."submissionStatus" = 'APPROVED'
JOIN "AgentGraph" a 
    ON slv."agentGraphId" = a.id 
    AND slv."agentGraphVersion" = a.version
LEFT JOIN "Profile" p 
    ON sl."owningUserId" = p."userId"
LEFT JOIN "mv_review_stats" rs 
    ON sl.id = rs."storeListingId"
LEFT JOIN "mv_agent_run_counts" ar 
    ON a.id = ar."agentGraphId"
LEFT JOIN agent_versions av 
    ON sl.id = av."storeListingId"
WHERE sl."isDeleted" = false
  AND sl."hasApprovedVersion" = true;

COMMIT;