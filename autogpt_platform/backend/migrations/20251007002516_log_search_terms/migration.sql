-- CreateTable
CREATE TABLE "SearchTerms" (
    "id" BIGSERIAL NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "searchTerm" TEXT NOT NULL,

    CONSTRAINT "SearchTerms_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "SearchTerms_createdAt_idx" ON "SearchTerms"("createdAt");
