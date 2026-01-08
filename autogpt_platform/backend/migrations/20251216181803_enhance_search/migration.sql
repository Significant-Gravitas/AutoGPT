-- DropIndex
DROP INDEX "StoreListingEmbedding_embedding_idx";

-- AlterTable
ALTER TABLE "StoreListingEmbedding" ALTER COLUMN "id" DROP DEFAULT;
