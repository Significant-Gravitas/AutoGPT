-- CreateEnum
CREATE TYPE "SubmissionStatus" AS ENUM ('DAFT', 'PENDING', 'APPROVED', 'REJECTED');

-- CreateTable
CREATE TABLE "Profile" (
    "id" UUID NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT NOT NULL,
    "username" TEXT NOT NULL,
    "description" TEXT NOT NULL,
    "links" TEXT[],
    "avatarUrl" TEXT,

    CONSTRAINT "Profile_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "StoreListing" (
    "id" UUID NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "isDeleted" BOOLEAN NOT NULL DEFAULT false,
    "isApproved" BOOLEAN NOT NULL DEFAULT false,
    "agentId" TEXT NOT NULL,
    "agentVersion" INTEGER NOT NULL,
    "owningUserId" TEXT NOT NULL,

    CONSTRAINT "StoreListing_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "StoreListingVersion" (
    "id" UUID NOT NULL,
    "version" INTEGER NOT NULL DEFAULT 1,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "agentId" TEXT NOT NULL,
    "agentVersion" INTEGER NOT NULL,
    "slug" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "videoUrl" TEXT,
    "imageUrls" TEXT[],
    "description" TEXT NOT NULL,
    "categories" TEXT[],
    "isFeatured" BOOLEAN NOT NULL DEFAULT false,
    "isDeleted" BOOLEAN NOT NULL DEFAULT false,
    "isAvailable" BOOLEAN NOT NULL DEFAULT true,
    "isApproved" BOOLEAN NOT NULL DEFAULT false,
    "storeListingId" UUID,

    CONSTRAINT "StoreListingVersion_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "StoreListingReview" (
    "id" UUID NOT NULL DEFAULT gen_random_uuid(),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "storeListingVersionId" UUID NOT NULL,
    "reviewByUserId" TEXT NOT NULL,
    "score" INTEGER NOT NULL,
    "comments" TEXT,

    CONSTRAINT "StoreListingReview_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "StoreListingSubmission" (
    "id" UUID NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "storeListingId" UUID NOT NULL,
    "storeListingVersionId" UUID NOT NULL,
    "reviewerId" TEXT NOT NULL,
    "Status" "SubmissionStatus" NOT NULL DEFAULT 'PENDING',
    "reviewComments" TEXT,

    CONSTRAINT "StoreListingSubmission_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "Profile_username_key" ON "Profile"("username");

-- CreateIndex
CREATE INDEX "Profile_username_idx" ON "Profile"("username");

-- CreateIndex
CREATE INDEX "StoreListing_isApproved_idx" ON "StoreListing"("isApproved");

-- CreateIndex
CREATE INDEX "StoreListing_agentId_idx" ON "StoreListing"("agentId");

-- CreateIndex
CREATE INDEX "StoreListing_owningUserId_idx" ON "StoreListing"("owningUserId");

-- CreateIndex
CREATE INDEX "StoreListingVersion_agentId_agentVersion_isApproved_idx" ON "StoreListingVersion"("agentId", "agentVersion", "isApproved");

-- CreateIndex
CREATE UNIQUE INDEX "StoreListingVersion_agentId_agentVersion_key" ON "StoreListingVersion"("agentId", "agentVersion");

-- CreateIndex
CREATE INDEX "StoreListingSubmission_storeListingId_idx" ON "StoreListingSubmission"("storeListingId");

-- CreateIndex
CREATE INDEX "StoreListingSubmission_Status_idx" ON "StoreListingSubmission"("Status");

-- AddForeignKey
ALTER TABLE "Profile" ADD CONSTRAINT "Profile_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "StoreListing" ADD CONSTRAINT "StoreListing_agentId_agentVersion_fkey" FOREIGN KEY ("agentId", "agentVersion") REFERENCES "AgentGraph"("id", "version") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "StoreListing" ADD CONSTRAINT "StoreListing_owningUserId_fkey" FOREIGN KEY ("owningUserId") REFERENCES "User"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "StoreListingVersion" ADD CONSTRAINT "StoreListingVersion_agentId_agentVersion_fkey" FOREIGN KEY ("agentId", "agentVersion") REFERENCES "AgentGraph"("id", "version") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "StoreListingVersion" ADD CONSTRAINT "StoreListingVersion_storeListingId_fkey" FOREIGN KEY ("storeListingId") REFERENCES "StoreListing"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "StoreListingReview" ADD CONSTRAINT "StoreListingReview_storeListingVersionId_fkey" FOREIGN KEY ("storeListingVersionId") REFERENCES "StoreListingVersion"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "StoreListingReview" ADD CONSTRAINT "StoreListingReview_reviewByUserId_fkey" FOREIGN KEY ("reviewByUserId") REFERENCES "User"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "StoreListingSubmission" ADD CONSTRAINT "StoreListingSubmission_storeListingId_fkey" FOREIGN KEY ("storeListingId") REFERENCES "StoreListing"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "StoreListingSubmission" ADD CONSTRAINT "StoreListingSubmission_storeListingVersionId_fkey" FOREIGN KEY ("storeListingVersionId") REFERENCES "StoreListingVersion"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "StoreListingSubmission" ADD CONSTRAINT "StoreListingSubmission_reviewerId_fkey" FOREIGN KEY ("reviewerId") REFERENCES "User"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
