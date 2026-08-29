/*
  Warnings:

  - You are about to drop the column `blockId` on the `CreditTransaction` table. All the data in the column will be moved to metadata->block_id.

*/
BEGIN;

-- DropForeignKey blockId
ALTER TABLE "CreditTransaction" DROP CONSTRAINT "CreditTransaction_blockId_fkey";

-- Update migrate blockId into metadata->"block_id"
UPDATE "CreditTransaction"
SET "metadata" = jsonb_set(
    COALESCE("metadata"::jsonb, '{}'),
    '{block_id}',
    to_jsonb("blockId")
)
WHERE "blockId" IS NOT NULL;

-- AlterTable drop blockId
ALTER TABLE "CreditTransaction" DROP COLUMN "blockId";

COMMIT;

/*
    These indices dropped below were part of the cleanup during the schema change applied above.
    These indexes were not useful and will not impact anything upon their removal.
*/

-- DropIndex
DROP INDEX "StoreListingReview_storeListingVersionId_idx";

-- DropIndex
DROP INDEX "StoreListingSubmission_Status_idx";
