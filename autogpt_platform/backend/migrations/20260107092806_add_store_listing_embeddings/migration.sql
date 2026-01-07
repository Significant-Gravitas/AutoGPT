-- CreateExtension
CREATE EXTENSION IF NOT EXISTS "vector" WITH SCHEMA "public";

-- CreateTable
CREATE TABLE "StoreListingEmbedding" (
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "storeListingVersionId" TEXT NOT NULL,
    "embedding" public.vector(1536),

    CONSTRAINT "StoreListingEmbedding_pkey" PRIMARY KEY ("storeListingVersionId")
);

-- AddForeignKey
ALTER TABLE "StoreListingEmbedding" ADD CONSTRAINT "StoreListingEmbedding_storeListingVersionId_fkey" FOREIGN KEY ("storeListingVersionId") REFERENCES "StoreListingVersion"("id") ON DELETE CASCADE ON UPDATE CASCADE;

CREATE INDEX idx_store_listing_embedding_hnsw ON "StoreListingEmbedding" USING hnsw (embedding public.vector_cosine_ops);
