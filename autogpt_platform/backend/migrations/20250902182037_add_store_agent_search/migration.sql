-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- CreateEnum
CREATE TYPE "SearchFieldType" AS ENUM ('NAME', 'DESCRIPTION', 'CATEGORIES', 'SUBHEADING');

-- CreateTable
CREATE TABLE "StoreAgentSearch" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "storeListingVersionId" TEXT NOT NULL,
    "storeListingId" TEXT NOT NULL,
    "fieldName" TEXT NOT NULL,
    "fieldValue" TEXT NOT NULL,
    "embedding" vector(1536),
    "fieldType" "SearchFieldType" NOT NULL,
    "submissionStatus" "SubmissionStatus" NOT NULL,
    "isAvailable" BOOLEAN NOT NULL,

    CONSTRAINT "StoreAgentSearch_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "StoreAgentSearch_storeListingVersionId_idx" ON "StoreAgentSearch"("storeListingVersionId");

-- CreateIndex
CREATE INDEX "StoreAgentSearch_storeListingId_idx" ON "StoreAgentSearch"("storeListingId");

-- CreateIndex
CREATE INDEX "StoreAgentSearch_fieldName_idx" ON "StoreAgentSearch"("fieldName");

-- CreateIndex
CREATE INDEX "StoreAgentSearch_fieldType_idx" ON "StoreAgentSearch"("fieldType");

-- CreateIndex
CREATE INDEX "StoreAgentSearch_submissionStatus_isAvailable_idx" ON "StoreAgentSearch"("submissionStatus", "isAvailable");

-- CreateIndex
CREATE UNIQUE INDEX "StoreAgentSearch_storeListingVersionId_fieldName_key" ON "StoreAgentSearch"("storeListingVersionId", "fieldName");

-- Create HNSW index for vector similarity search (pgvector)
CREATE INDEX "StoreAgentSearch_embedding_idx" ON "StoreAgentSearch" USING hnsw (embedding vector_cosine_ops);

-- AddForeignKey
ALTER TABLE "StoreAgentSearch" ADD CONSTRAINT "StoreAgentSearch_storeListingVersionId_fkey" FOREIGN KEY ("storeListingVersionId") REFERENCES "StoreListingVersion"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "StoreAgentSearch" ADD CONSTRAINT "StoreAgentSearch_storeListingId_fkey" FOREIGN KEY ("storeListingId") REFERENCES "StoreListing"("id") ON DELETE CASCADE ON UPDATE CASCADE;
