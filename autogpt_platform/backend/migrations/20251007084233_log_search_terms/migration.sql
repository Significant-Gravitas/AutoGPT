-- CreateTable
CREATE TABLE "SearchTerms" (
    "id" BIGSERIAL NOT NULL,
    "createdDate" TIMESTAMP(3) NOT NULL,
    "searchTerm" TEXT NOT NULL,

    CONSTRAINT "SearchTerms_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "SearchTerms_createdDate_idx" ON "SearchTerms"("createdDate");
