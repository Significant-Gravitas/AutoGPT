/*
  Warnings:

  - You are about to drop the column `search` on the `StoreListingVersion` table. All the data in the column will be lost.

*/
-- AlterEnum
ALTER TYPE "OnboardingStep" ADD VALUE 'VISIT_COPILOT';

-- DropIndex
DROP INDEX "UnifiedContentEmbedding_search_idx";

-- AlterTable
ALTER TABLE "StoreListingVersion" DROP COLUMN "search";
