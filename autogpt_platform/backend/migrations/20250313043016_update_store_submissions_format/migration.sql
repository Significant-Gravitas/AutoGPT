/*
  Warnings:

  - The values [DAFT] on the enum `SubmissionStatus` will be removed. If these variants are still used in the database, this will fail.
  - You are about to drop the column `isApproved` on the `StoreListing` table. All the data in the column will be lost.
  - You are about to drop the column `slug` on the `StoreListingVersion` table. All the data in the column will be lost.
  - You are about to drop the `StoreListingSubmission` table. If the table is not empty, all the data it contains will be lost.
  - A unique constraint covering the columns `[activeVersionId]` on the table `StoreListing` will be added. If there are existing duplicate values, this will fail.
  - A unique constraint covering the columns `[storeListingId,version]` on the table `StoreListingVersion` will be added. If there are existing duplicate values, this will fail.
  - Made the column `storeListingId` on table `StoreListingVersion` required. This step will fail if there are existing NULL values in that column.

*/

-- First, drop all views that depend on the columns we're modifying
DROP VIEW IF EXISTS platform."StoreSubmission";
DROP VIEW IF EXISTS platform."StoreAgent";
DROP VIEW IF EXISTS platform."Creator";

-- Then proceed with the table changes
BEGIN;
-- Create the new enum type
CREATE TYPE "SubmissionStatus_new" AS ENUM ('DRAFT', 'PENDING', 'APPROVED', 'REJECTED');

-- Modify the column with the correct casing (Status with capital S)
ALTER TABLE "StoreListingSubmission" ALTER COLUMN "Status" DROP DEFAULT;
ALTER TABLE "StoreListingSubmission" ALTER COLUMN "Status" TYPE "SubmissionStatus_new" 
    USING (CASE WHEN "Status"::text = 'DAFT' THEN 'DRAFT'::text 
          ELSE "Status"::text 
          END)::"SubmissionStatus_new";

-- Rename the enum types
ALTER TYPE "SubmissionStatus" RENAME TO "SubmissionStatus_old";
ALTER TYPE "SubmissionStatus_new" RENAME TO "SubmissionStatus";
DROP TYPE "SubmissionStatus_old";

-- Set default back
ALTER TABLE "StoreListingSubmission" ALTER COLUMN "Status" SET DEFAULT 'PENDING';
COMMIT;

-- Drop constraints
ALTER TABLE "StoreListingSubmission" DROP CONSTRAINT IF EXISTS "StoreListingSubmission_reviewerId_fkey";

-- Drop indexes
DROP INDEX IF EXISTS "StoreListing_isDeleted_isApproved_idx";
DROP INDEX IF EXISTS "StoreListingSubmission_storeListingVersionId_key";

-- Modify StoreListing
ALTER TABLE "StoreListing" 
DROP COLUMN IF EXISTS "isApproved",
ADD COLUMN IF NOT EXISTS "activeVersionId" TEXT,
ADD COLUMN IF NOT EXISTS "hasApprovedVersion" BOOLEAN NOT NULL DEFAULT false,
ADD COLUMN IF NOT EXISTS "slug" TEXT;

-- First add ALL columns to StoreListingVersion (including the submissionStatus column)
ALTER TABLE "StoreListingVersion" 
ADD COLUMN IF NOT EXISTS "reviewerId" TEXT,
ADD COLUMN IF NOT EXISTS "reviewComments" TEXT,
ADD COLUMN IF NOT EXISTS "internalComments" TEXT,
ADD COLUMN IF NOT EXISTS "reviewedAt" TIMESTAMP(3),
ADD COLUMN IF NOT EXISTS "changesSummary" TEXT,
ADD COLUMN IF NOT EXISTS "approvedAt" TIMESTAMP(3),
ADD COLUMN IF NOT EXISTS "rejectedAt" TIMESTAMP(3),
ADD COLUMN IF NOT EXISTS "submissionStatus" "SubmissionStatus" NOT NULL DEFAULT 'DRAFT',
ADD COLUMN IF NOT EXISTS "submittedAt" TIMESTAMP(3),
ALTER COLUMN "storeListingId" SET NOT NULL,
ALTER COLUMN "name" DROP NOT NULL;

-- NOW copy data from StoreListingSubmission to StoreListingVersion
DO $$
BEGIN
    -- First, check what columns actually exist in the StoreListingSubmission table
    DECLARE
        has_reviewerId BOOLEAN := (SELECT EXISTS (
            SELECT FROM information_schema.columns 
            WHERE table_name = 'StoreListingSubmission' 
            AND column_name = 'reviewerId'
        ));
        
        has_reviewComments BOOLEAN := (SELECT EXISTS (
            SELECT FROM information_schema.columns 
            WHERE table_name = 'StoreListingSubmission' 
            AND column_name = 'reviewComments'
        ));
        
        has_changesSummary BOOLEAN := (SELECT EXISTS (
            SELECT FROM information_schema.columns 
            WHERE table_name = 'StoreListingSubmission' 
            AND column_name = 'changesSummary'
        ));
    BEGIN
        -- Only copy fields that we know exist
        IF has_reviewerId THEN
            UPDATE "StoreListingVersion" AS v
            SET "reviewerId" = s."reviewerId"
            FROM "StoreListingSubmission" AS s
            WHERE v."id" = s."storeListingVersionId";
        END IF;
        
        IF has_reviewComments THEN
            UPDATE "StoreListingVersion" AS v
            SET "reviewComments" = s."reviewComments"
            FROM "StoreListingSubmission" AS s
            WHERE v."id" = s."storeListingVersionId";
        END IF;
        
        IF has_changesSummary THEN
            UPDATE "StoreListingVersion" AS v
            SET "changesSummary" = s."changesSummary"
            FROM "StoreListingSubmission" AS s
            WHERE v."id" = s."storeListingVersionId";
        END IF;
    END;
    
    -- Update submission status based on StoreListingSubmission status
    UPDATE "StoreListingVersion" AS v
    SET "submissionStatus" = s."Status"
    FROM "StoreListingSubmission" AS s
    WHERE v."id" = s."storeListingVersionId";
END
$$;

-- Drop the StoreListingSubmission table
DROP TABLE IF EXISTS "StoreListingSubmission";

-- Copy slugs from StoreListingVersion to StoreListing
WITH latest_versions AS (
    SELECT 
        "storeListingId",
        "slug",
        ROW_NUMBER() OVER (PARTITION BY "storeListingId" ORDER BY "version" DESC) as rn
    FROM "StoreListingVersion"
)
UPDATE "StoreListing" sl
SET "slug" = lv."slug"
FROM latest_versions lv
WHERE sl."id" = lv."storeListingId" AND lv.rn = 1;

-- Make StoreListing.slug required and unique
ALTER TABLE "StoreListing" ALTER COLUMN "slug" SET NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS "StoreListing_slug_key" ON "StoreListing"("slug");

-- Drop the slug column from StoreListingVersion since it's now on StoreListing
ALTER TABLE "StoreListingVersion" DROP COLUMN "slug";

-- Drop the unique constraint on agentId, agentVersion in StoreListingVersion
ALTER TABLE "StoreListingVersion" DROP CONSTRAINT IF EXISTS "StoreListingVersion_agentId_agentVersion_key";


-- Create new indexes
CREATE UNIQUE INDEX IF NOT EXISTS "StoreListing_activeVersionId_key" 
ON "StoreListing"("activeVersionId");

CREATE INDEX IF NOT EXISTS "StoreListing_isDeleted_hasApprovedVersion_idx" 
ON "StoreListing"("isDeleted", "hasApprovedVersion");

CREATE INDEX IF NOT EXISTS "StoreListingVersion_storeListingId_isApproved_isAvailable_idx" 
ON "StoreListingVersion"("storeListingId", "isApproved", "isAvailable");

CREATE INDEX IF NOT EXISTS "StoreListingVersion_submissionStatus_idx" 
ON "StoreListingVersion"("submissionStatus");

CREATE UNIQUE INDEX IF NOT EXISTS "StoreListingVersion_storeListingId_version_key" 
ON "StoreListingVersion"("storeListingId", "version");

-- Add foreign keys
ALTER TABLE "StoreListing" 
ADD CONSTRAINT "StoreListing_activeVersionId_fkey" 
FOREIGN KEY ("activeVersionId") REFERENCES "StoreListingVersion"("id") 
ON DELETE SET NULL ON UPDATE CASCADE;

-- Add reviewer foreign key
ALTER TABLE "StoreListingVersion" 
ADD CONSTRAINT "StoreListingVersion_reviewerId_fkey" 
FOREIGN KEY ("reviewerId") REFERENCES "User"("id") 
ON DELETE SET NULL ON UPDATE CASCADE;

-- Add index for reviewer
CREATE INDEX IF NOT EXISTS "StoreListingVersion_reviewerId_idx" 
ON "StoreListingVersion"("reviewerId");

-- Recreate the views with updated column references

-- 1. Recreate StoreSubmission view
CREATE VIEW platform."StoreSubmission" AS  
SELECT 
    sl.id AS listing_id,
    sl."owningUserId" AS user_id,
    slv."agentId" AS agent_id,
    slv.version AS agent_version,
    sl.slug,
    COALESCE(slv.name, '') AS name,
    slv."subHeading" AS sub_heading,
    slv.description,
    slv."imageUrls" AS image_urls,
    slv."submittedAt" AS date_submitted,
    slv."submissionStatus" AS status,
    COALESCE(ar.run_count, 0::bigint) AS runs,
    COALESCE(avg(sr.score::numeric), 0.0)::double precision AS rating
FROM platform."StoreListing" sl
    JOIN platform."StoreListingVersion" slv ON slv."storeListingId" = sl.id
    LEFT JOIN platform."StoreListingReview" sr ON sr."storeListingVersionId" = slv.id
    LEFT JOIN (
        SELECT "AgentGraphExecution"."agentGraphId", count(*) AS run_count
        FROM platform."AgentGraphExecution"
        GROUP BY "AgentGraphExecution"."agentGraphId"
    ) ar ON ar."agentGraphId" = slv."agentId"
WHERE sl."isDeleted" = false
GROUP BY sl.id, sl."owningUserId", slv."agentId", slv.version, sl.slug, slv.name, 
    slv."subHeading", slv.description, slv."imageUrls", slv."submittedAt", slv."submissionStatus", ar.run_count;

-- 2. Recreate StoreAgent view
CREATE VIEW platform."StoreAgent" AS  
WITH reviewstats AS (
    SELECT sl_1.id AS "storeListingId",
        count(sr.id) AS review_count,
        avg(sr.score::numeric) AS avg_rating
    FROM platform."StoreListing" sl_1
        JOIN platform."StoreListingVersion" slv_1 ON slv_1."storeListingId" = sl_1.id
        JOIN platform."StoreListingReview" sr ON sr."storeListingVersionId" = slv_1.id
    WHERE sl_1."isDeleted" = false
    GROUP BY sl_1.id
), agentruns AS (
    SELECT "AgentGraphExecution"."agentGraphId",
        count(*) AS run_count
    FROM platform."AgentGraphExecution"
    GROUP BY "AgentGraphExecution"."agentGraphId"
)
SELECT sl.id AS listing_id,
    slv.id AS "storeListingVersionId",
    slv."createdAt" AS updated_at,
    sl.slug,
    COALESCE(slv.name, '') AS agent_name,
    slv."videoUrl" AS agent_video,
    COALESCE(slv."imageUrls", ARRAY[]::text[]) AS agent_image,
    slv."isFeatured" AS featured,
    p.username AS creator_username,
    p."avatarUrl" AS creator_avatar,
    slv."subHeading" AS sub_heading,
    slv.description,
    slv.categories,
    COALESCE(ar.run_count, 0::bigint) AS runs,
    COALESCE(rs.avg_rating, 0.0)::double precision AS rating,
    array_agg(DISTINCT slv.version::text) AS versions
FROM platform."StoreListing" sl
    JOIN platform."AgentGraph" a ON sl."agentId" = a.id AND sl."agentVersion" = a.version
    LEFT JOIN platform."Profile" p ON sl."owningUserId" = p."userId"
    LEFT JOIN platform."StoreListingVersion" slv ON slv."storeListingId" = sl.id
    LEFT JOIN reviewstats rs ON sl.id = rs."storeListingId"
    LEFT JOIN agentruns ar ON a.id = ar."agentGraphId"
WHERE sl."isDeleted" = false AND sl."hasApprovedVersion" = true  -- Changed from isApproved to hasApprovedVersion
GROUP BY sl.id, slv.id, sl.slug, slv."createdAt", slv.name, slv."videoUrl", slv."imageUrls", 
    slv."isFeatured", p.username, p."avatarUrl", slv."subHeading", slv.description, 
    slv.categories, ar.run_count, rs.avg_rating;

-- 3. Recreate Creator view
CREATE VIEW platform."Creator" AS  
WITH agentstats AS (
    SELECT p_1.username,
        count(DISTINCT sl.id) AS num_agents,
        avg(COALESCE(sr.score, 0)::numeric) AS agent_rating,
        sum(COALESCE(age.run_count, 0::bigint)) AS agent_runs
    FROM platform."Profile" p_1
        LEFT JOIN platform."StoreListing" sl ON sl."owningUserId" = p_1."userId"
        LEFT JOIN platform."StoreListingVersion" slv ON slv."storeListingId" = sl.id
        LEFT JOIN platform."StoreListingReview" sr ON sr."storeListingVersionId" = slv.id
        LEFT JOIN (
            SELECT "AgentGraphExecution"."agentGraphId", count(*) AS run_count
            FROM platform."AgentGraphExecution"
            GROUP BY "AgentGraphExecution"."agentGraphId"
        ) age ON age."agentGraphId" = sl."agentId"
    WHERE sl."isDeleted" = false AND sl."hasApprovedVersion" = true  -- Changed from isApproved to hasApprovedVersion
    GROUP BY p_1.username
)
SELECT p.username,
    p.name,
    p."avatarUrl" AS avatar_url,
    p.description,
    array_agg(DISTINCT cats.c) FILTER (WHERE cats.c IS NOT NULL) AS top_categories,
    p.links,
    p."isFeatured" AS is_featured,
    COALESCE(ast.num_agents, 0::bigint) AS num_agents,
    COALESCE(ast.agent_rating, 0.0) AS agent_rating,
    COALESCE(ast.agent_runs, 0::numeric) AS agent_runs
FROM platform."Profile" p
    LEFT JOIN agentstats ast ON ast.username = p.username
    LEFT JOIN LATERAL (
        SELECT unnest(slv.categories) AS c
        FROM platform."StoreListing" sl
            JOIN platform."StoreListingVersion" slv ON slv."storeListingId" = sl.id
        WHERE sl."owningUserId" = p."userId" AND sl."isDeleted" = false AND sl."hasApprovedVersion" = true  -- Changed from isApproved to hasApprovedVersion
    ) cats ON true
GROUP BY p.username, p.name, p."avatarUrl", p.description, p.links, p."isFeatured", ast.num_agents, ast.agent_rating, ast.agent_runs;