-- CreateEnum
CREATE TYPE "SubmissionStatus" AS ENUM ('DAFT', 'PENDING', 'APPROVED', 'REJECTED');

-- AlterTable
ALTER TABLE "AgentGraphExecution" ADD COLUMN     "agentPresetId" TEXT;

-- AlterTable
ALTER TABLE "AgentNodeExecutionInputOutput" ADD COLUMN     "agentPresetId" TEXT;

-- AlterTable
ALTER TABLE "AnalyticsMetrics" ALTER COLUMN "id" DROP DEFAULT;

-- AlterTable
ALTER TABLE "CreditTransaction" RENAME CONSTRAINT "UserBlockCredit_pkey" TO "CreditTransaction_pkey";

-- CreateTable
CREATE TABLE "AgentPreset" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "name" TEXT NOT NULL,
    "description" TEXT NOT NULL,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "userId" TEXT NOT NULL,
    "agentId" TEXT NOT NULL,
    "agentVersion" INTEGER NOT NULL,

    CONSTRAINT "AgentPreset_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "UserAgent" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT NOT NULL,
    "agentId" TEXT NOT NULL,
    "agentVersion" INTEGER NOT NULL,
    "agentPresetId" TEXT,
    "isFavorite" BOOLEAN NOT NULL DEFAULT false,
    "isCreatedByUser" BOOLEAN NOT NULL DEFAULT false,
    "isArchived" BOOLEAN NOT NULL DEFAULT false,
    "isDeleted" BOOLEAN NOT NULL DEFAULT false,

    CONSTRAINT "UserAgent_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Profile" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT,
    "name" TEXT NOT NULL,
    "username" TEXT NOT NULL,
    "description" TEXT NOT NULL,
    "links" TEXT[],
    "avatarUrl" TEXT,

    CONSTRAINT "Profile_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "StoreListing" (
    "id" TEXT NOT NULL,
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
    "id" TEXT NOT NULL,
    "version" INTEGER NOT NULL DEFAULT 1,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "agentId" TEXT NOT NULL,
    "agentVersion" INTEGER NOT NULL,
    "slug" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "subHeading" TEXT NOT NULL,
    "videoUrl" TEXT,
    "imageUrls" TEXT[],
    "description" TEXT NOT NULL,
    "categories" TEXT[],
    "isFeatured" BOOLEAN NOT NULL DEFAULT false,
    "isDeleted" BOOLEAN NOT NULL DEFAULT false,
    "isAvailable" BOOLEAN NOT NULL DEFAULT true,
    "isApproved" BOOLEAN NOT NULL DEFAULT false,
    "storeListingId" TEXT,

    CONSTRAINT "StoreListingVersion_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "StoreListingReview" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "storeListingVersionId" TEXT NOT NULL,
    "reviewByUserId" TEXT NOT NULL,
    "score" INTEGER NOT NULL,
    "comments" TEXT,

    CONSTRAINT "StoreListingReview_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "StoreListingSubmission" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "storeListingId" TEXT NOT NULL,
    "storeListingVersionId" TEXT NOT NULL,
    "reviewerId" TEXT NOT NULL,
    "Status" "SubmissionStatus" NOT NULL DEFAULT 'PENDING',
    "reviewComments" TEXT,

    CONSTRAINT "StoreListingSubmission_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "AgentPreset_userId_idx" ON "AgentPreset"("userId");

-- CreateIndex
CREATE INDEX "UserAgent_userId_idx" ON "UserAgent"("userId");

-- CreateIndex
CREATE UNIQUE INDEX "Profile_username_key" ON "Profile"("username");

-- CreateIndex
CREATE INDEX "Profile_username_idx" ON "Profile"("username");

-- CreateIndex
CREATE INDEX "Profile_userId_idx" ON "Profile"("userId");

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
CREATE INDEX "StoreListingReview_storeListingVersionId_idx" ON "StoreListingReview"("storeListingVersionId");

-- CreateIndex
CREATE UNIQUE INDEX "StoreListingReview_storeListingVersionId_reviewByUserId_key" ON "StoreListingReview"("storeListingVersionId", "reviewByUserId");

-- CreateIndex
CREATE INDEX "StoreListingSubmission_storeListingId_idx" ON "StoreListingSubmission"("storeListingId");

-- CreateIndex
CREATE INDEX "StoreListingSubmission_Status_idx" ON "StoreListingSubmission"("Status");

-- RenameForeignKey
ALTER TABLE "CreditTransaction" RENAME CONSTRAINT "UserBlockCredit_blockId_fkey" TO "CreditTransaction_blockId_fkey";

-- RenameForeignKey
ALTER TABLE "CreditTransaction" RENAME CONSTRAINT "UserBlockCredit_userId_fkey" TO "CreditTransaction_userId_fkey";

-- AddForeignKey
ALTER TABLE "AgentPreset" ADD CONSTRAINT "AgentPreset_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AgentPreset" ADD CONSTRAINT "AgentPreset_agentId_agentVersion_fkey" FOREIGN KEY ("agentId", "agentVersion") REFERENCES "AgentGraph"("id", "version") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "UserAgent" ADD CONSTRAINT "UserAgent_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "UserAgent" ADD CONSTRAINT "UserAgent_agentId_agentVersion_fkey" FOREIGN KEY ("agentId", "agentVersion") REFERENCES "AgentGraph"("id", "version") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "UserAgent" ADD CONSTRAINT "UserAgent_agentPresetId_fkey" FOREIGN KEY ("agentPresetId") REFERENCES "AgentPreset"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AgentGraphExecution" ADD CONSTRAINT "AgentGraphExecution_agentPresetId_fkey" FOREIGN KEY ("agentPresetId") REFERENCES "AgentPreset"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AgentNodeExecutionInputOutput" ADD CONSTRAINT "AgentNodeExecutionInputOutput_agentPresetId_fkey" FOREIGN KEY ("agentPresetId") REFERENCES "AgentPreset"("id") ON DELETE SET NULL ON UPDATE CASCADE;

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

-- RenameIndex
ALTER INDEX "UserBlockCredit_userId_createdAt_idx" RENAME TO "CreditTransaction_userId_createdAt_idx";
