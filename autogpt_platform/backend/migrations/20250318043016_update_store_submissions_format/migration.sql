/*
  Warnings:

  - The enum type "SubmissionStatus" will be replaced. The 'DAFT' value is removed, so any data using 'DAFT' will be updated to 'DRAFT'. If there are rows still expecting 'DAFT' after this change, it will fail.
  - You are about to drop the column "isApproved" on the "StoreListing" table. All the data in that column will be lost.
  - You are about to drop the column "slug" on the "StoreListingVersion" table. All the data in that column will be lost.
  - You are about to drop the "StoreListingSubmission" table. Data in that table (beyond what is copied over) will be permanently lost.
  - A unique constraint covering the column "activeVersionId" on the "StoreListing" table will be added. If duplicates already exist, this will fail.
  - A unique constraint covering the columns ("storeListingId","version") on "StoreListingVersion" will be added. If duplicates already exist, this will fail.
  - The "storeListingId" column on "StoreListingVersion" is set to NOT NULL. If any rows currently have a NULL value, this step will fail.
  - The views "StoreSubmission", "StoreAgent", and "Creator" are dropped and recreated. Any usage or references to them will be momentarily disrupted until the views are recreated.
*/

BEGIN;

-- First, drop all views that depend on the columns and types we're modifying
DROP VIEW IF EXISTS "StoreSubmission";
DROP VIEW IF EXISTS "StoreAgent";
DROP VIEW IF EXISTS "Creator";

-- Create the new enum type
CREATE TYPE "SubmissionStatus_new" AS ENUM ('DRAFT', 'PENDING', 'APPROVED', 'REJECTED');

-- Modify the column with the correct casing (Status with capital S)
ALTER TABLE "StoreListingSubmission" ALTER COLUMN "Status" DROP DEFAULT;
ALTER TABLE "StoreListingSubmission"
    ALTER COLUMN "Status" TYPE "SubmissionStatus_new"
    USING (
      CASE WHEN "Status"::text = 'DAFT' THEN 'DRAFT'::text
           ELSE "Status"::text
      END
    )::"SubmissionStatus_new";

-- Rename the enum types
ALTER TYPE "SubmissionStatus" RENAME TO "SubmissionStatus_old";
ALTER TYPE "SubmissionStatus_new" RENAME TO "SubmissionStatus";
DROP TYPE "SubmissionStatus_old";

-- Set default back
ALTER TABLE "StoreListingSubmission" ALTER COLUMN "Status" SET DEFAULT 'PENDING';

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
    ADD COLUMN IF NOT EXISTS "submissionStatus" "SubmissionStatus" NOT NULL DEFAULT 'DRAFT',
    ADD COLUMN IF NOT EXISTS "submittedAt" TIMESTAMP(3),
    ALTER COLUMN "storeListingId" SET NOT NULL;

-- NOW copy data from StoreListingSubmission to StoreListingVersion
DO $$
BEGIN
    -- First, check what columns actually exist in the StoreListingSubmission table
    DECLARE
        has_reviewerId BOOLEAN := (
            SELECT EXISTS (
                SELECT FROM information_schema.columns
                WHERE table_name = 'StoreListingSubmission'
                AND column_name = 'reviewerId'
            )
        );

        has_reviewComments BOOLEAN := (
            SELECT EXISTS (
                SELECT FROM information_schema.columns
                WHERE table_name = 'StoreListingSubmission'
                AND column_name = 'reviewComments'
            )
        );

        has_changesSummary BOOLEAN := (
            SELECT EXISTS (
                SELECT FROM information_schema.columns
                WHERE table_name = 'StoreListingSubmission'
                AND column_name = 'changesSummary'
            )
        );
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

    -- Update reviewedAt timestamps for versions with APPROVED or REJECTED status
    UPDATE "StoreListingVersion" AS v
    SET "reviewedAt" = s."updatedAt"
    FROM "StoreListingSubmission" AS s
    WHERE v."id" = s."storeListingVersionId"
      AND s."Status" IN ('APPROVED', 'REJECTED');
END;
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
WHERE sl."id" = lv."storeListingId"
  AND lv.rn = 1;

-- Make StoreListing.slug required and unique
ALTER TABLE "StoreListing" ALTER COLUMN "slug" SET NOT NULL;
CREATE UNIQUE INDEX "StoreListing_owningUserId_slug_key" ON "StoreListing"("owningUserId", "slug");
DROP INDEX "StoreListing_owningUserId_idx";

-- Drop the slug column from StoreListingVersion since it's now on StoreListing
ALTER TABLE "StoreListingVersion" DROP COLUMN "slug";

-- Update both sides of the relation from one-to-one to one-to-many
-- The AgentGraph->StoreListingVersion relationship is now one-to-many

-- Drop the unique constraint but add a non-unique index for query performance
ALTER TABLE "StoreListingVersion" DROP CONSTRAINT IF EXISTS "StoreListingVersion_agentId_agentVersion_key";
CREATE INDEX IF NOT EXISTS "StoreListingVersion_agentId_agentVersion_idx"
    ON "StoreListingVersion"("agentId", "agentVersion");

-- Set isApproved based on submissionStatus before removing it
UPDATE "StoreListingVersion"
SET "submissionStatus" = 'APPROVED'
WHERE "isApproved" = true;

-- Drop the isApproved column from StoreListingVersion since it's redundant with submissionStatus
ALTER TABLE "StoreListingVersion" DROP COLUMN "isApproved";

-- Initialize hasApprovedVersion for existing StoreListing rows ***
-- This sets "hasApprovedVersion" = TRUE for any StoreListing
-- that has at least one corresponding version with "APPROVED" status.
UPDATE "StoreListing" sl
SET "hasApprovedVersion" = (
  SELECT COUNT(*) > 0
  FROM "StoreListingVersion" slv
  WHERE slv."storeListingId" = sl.id
    AND slv."submissionStatus" = 'APPROVED'
    AND sl."agentId" = slv."agentId"
    AND sl."agentVersion" = slv."agentVersion"
);

-- Create new indexes
CREATE UNIQUE INDEX IF NOT EXISTS "StoreListing_activeVersionId_key"
    ON "StoreListing"("activeVersionId");

CREATE INDEX IF NOT EXISTS "StoreListing_isDeleted_hasApprovedVersion_idx"
    ON "StoreListing"("isDeleted", "hasApprovedVersion");

CREATE INDEX IF NOT EXISTS "StoreListingVersion_storeListingId_submissionStatus_isAvailable_idx"
    ON "StoreListingVersion"("storeListingId", "submissionStatus", "isAvailable");

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

-- DropIndex
DROP INDEX "StoreListingVersion_agentId_agentVersion_key";

-- RenameIndex
ALTER INDEX "StoreListingVersion_storeListingId_submissionStatus_isAvailable_idx"
RENAME TO "StoreListingVersion_storeListingId_submissionStatus_isAvail_idx";

-- Recreate the views with updated column references

-- 1. Recreate StoreSubmission view
CREATE VIEW "StoreSubmission" AS
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
    COALESCE(avg(sr.score::numeric), 0.0)::double precision AS rating,
    -- Add the additional fields needed by the Pydantic model
    slv.id AS store_listing_version_id,
    slv."reviewerId" AS reviewer_id,
    slv."reviewComments" AS review_comments,
    slv."internalComments" AS internal_comments,
    slv."reviewedAt" AS reviewed_at,
    slv."changesSummary" AS changes_summary
FROM "StoreListing" sl
    JOIN "StoreListingVersion" slv ON slv."storeListingId" = sl.id
    LEFT JOIN "StoreListingReview" sr ON sr."storeListingVersionId" = slv.id
    LEFT JOIN (
        SELECT "AgentGraphExecution"."agentGraphId", count(*) AS run_count
        FROM "AgentGraphExecution"
        GROUP BY "AgentGraphExecution"."agentGraphId"
    ) ar ON ar."agentGraphId" = slv."agentId"
WHERE sl."isDeleted" = false
GROUP BY sl.id, sl."owningUserId", slv.id, slv."agentId", slv.version, sl.slug, slv.name,
         slv."subHeading", slv.description, slv."imageUrls", slv."submittedAt",
         slv."submissionStatus", slv."reviewerId", slv."reviewComments", slv."internalComments",
         slv."reviewedAt", slv."changesSummary", ar.run_count;

-- 2. Recreate StoreAgent view
CREATE VIEW "StoreAgent" AS
WITH reviewstats AS (
    SELECT sl_1.id AS "storeListingId",
           count(sr.id) AS review_count,
           avg(sr.score::numeric) AS avg_rating
      FROM "StoreListing" sl_1
      JOIN "StoreListingVersion" slv_1
           ON slv_1."storeListingId" = sl_1.id
      JOIN "StoreListingReview" sr
           ON sr."storeListingVersionId" = slv_1.id
     WHERE sl_1."isDeleted" = false
     GROUP BY sl_1.id
), agentruns AS (
    SELECT "AgentGraphExecution"."agentGraphId",
           count(*) AS run_count
      FROM "AgentGraphExecution"
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
  FROM "StoreListing" sl
  JOIN "AgentGraph" a
       ON sl."agentId" = a.id
      AND sl."agentVersion" = a.version
  LEFT JOIN "Profile" p
       ON sl."owningUserId" = p."userId"
  LEFT JOIN "StoreListingVersion" slv
       ON slv."storeListingId" = sl.id
  LEFT JOIN reviewstats rs
       ON sl.id = rs."storeListingId"
  LEFT JOIN agentruns ar
       ON a.id = ar."agentGraphId"
 WHERE sl."isDeleted" = false
   AND sl."hasApprovedVersion" = true
   AND slv."submissionStatus" = 'APPROVED'
 GROUP BY sl.id, slv.id, sl.slug, slv."createdAt", slv.name, slv."videoUrl",
          slv."imageUrls", slv."isFeatured", p.username, p."avatarUrl",
          slv."subHeading", slv.description, slv.categories, ar.run_count,
          rs.avg_rating;

-- 3. Recreate Creator view
CREATE VIEW "Creator" AS
WITH agentstats AS (
    SELECT p_1.username,
           count(DISTINCT sl.id) AS num_agents,
           avg(COALESCE(sr.score, 0)::numeric) AS agent_rating,
           sum(COALESCE(age.run_count, 0::bigint)) AS agent_runs
      FROM "Profile" p_1
 LEFT JOIN "StoreListing" sl
        ON sl."owningUserId" = p_1."userId"
 LEFT JOIN "StoreListingVersion" slv
        ON slv."storeListingId" = sl.id
 LEFT JOIN "StoreListingReview" sr
        ON sr."storeListingVersionId" = slv.id
 LEFT JOIN (
          SELECT "AgentGraphExecution"."agentGraphId",
                 count(*) AS run_count
            FROM "AgentGraphExecution"
        GROUP BY "AgentGraphExecution"."agentGraphId"
      ) age ON age."agentGraphId" = sl."agentId"
     WHERE sl."isDeleted" = false
       AND sl."hasApprovedVersion" = true
       AND slv."submissionStatus" = 'APPROVED'
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
  FROM "Profile" p
  LEFT JOIN agentstats ast
         ON ast.username = p.username
  LEFT JOIN LATERAL (
          SELECT unnest(slv.categories) AS c
            FROM "StoreListing" sl
            JOIN "StoreListingVersion" slv
                 ON slv."storeListingId" = sl.id
           WHERE sl."owningUserId" = p."userId"
             AND sl."isDeleted" = false
             AND sl."hasApprovedVersion" = true
             AND slv."submissionStatus" = 'APPROVED'
       ) cats ON true
 GROUP BY p.username, p.name, p."avatarUrl", p.description, p.links,
          p."isFeatured", ast.num_agents, ast.agent_rating, ast.agent_runs;

COMMIT;
