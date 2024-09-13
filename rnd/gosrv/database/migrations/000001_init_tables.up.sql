-- CreateEnum
CREATE TYPE "SubmissionStatus" AS ENUM ('PENDING', 'APPROVED', 'REJECTED');

-- CreateEnum
CREATE TYPE "InstallationLocation" AS ENUM ('LOCAL', 'CLOUD');

-- CreateTable
CREATE TABLE "Agents" (
    "id" UUID NOT NULL DEFAULT gen_random_uuid(),
    "version" INTEGER NOT NULL DEFAULT 1,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "submissionDate" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "submissionReviewDate" TIMESTAMP(3),
    "submissionStatus" "SubmissionStatus" NOT NULL DEFAULT 'PENDING',
    "submissionReviewComments" TEXT,
    "name" TEXT,
    "description" TEXT,
    "author" TEXT,
    "keywords" TEXT[],
    "categories" TEXT[],
    "search" tsvector DEFAULT ''::tsvector,
    "graph" JSONB NOT NULL,

    CONSTRAINT "Agents_pkey" PRIMARY KEY ("id","version")
);

-- CreateTable
CREATE TABLE "AnalyticsTracker" (
    "id" UUID NOT NULL DEFAULT gen_random_uuid(),
    "agentId" UUID NOT NULL,
    "views" INTEGER NOT NULL,
    "downloads" INTEGER NOT NULL,

    CONSTRAINT "AnalyticsTracker_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "InstallTracker" (
    "id" UUID NOT NULL DEFAULT gen_random_uuid(),
    "marketplaceAgentId" UUID NOT NULL,
    "installedAgentId" UUID NOT NULL,
    "installationLocation" "InstallationLocation" NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "InstallTracker_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "FeaturedAgent" (
    "id" UUID NOT NULL DEFAULT gen_random_uuid(),
    "agentId" UUID NOT NULL,
    "isActive" BOOLEAN NOT NULL DEFAULT false,
    "featuredCategories" TEXT[],
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "FeaturedAgent_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "Agents_id_key" ON "Agents"("id");

-- CreateIndex
CREATE UNIQUE INDEX "AnalyticsTracker_id_key" ON "AnalyticsTracker"("id");

-- CreateIndex
CREATE UNIQUE INDEX "AnalyticsTracker_agentId_key" ON "AnalyticsTracker"("agentId");

-- CreateIndex
CREATE UNIQUE INDEX "InstallTracker_marketplaceAgentId_installedAgentId_key" ON "InstallTracker"("marketplaceAgentId", "installedAgentId");

-- CreateIndex
CREATE UNIQUE INDEX "FeaturedAgent_id_key" ON "FeaturedAgent"("id");

-- CreateIndex
CREATE UNIQUE INDEX "FeaturedAgent_agentId_key" ON "FeaturedAgent"("agentId");

-- AddForeignKey
ALTER TABLE "AnalyticsTracker" ADD CONSTRAINT "AnalyticsTracker_agentId_fkey" FOREIGN KEY ("agentId") REFERENCES "Agents"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "InstallTracker" ADD CONSTRAINT "InstallTracker_marketplaceAgentId_fkey" FOREIGN KEY ("marketplaceAgentId") REFERENCES "Agents"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "FeaturedAgent" ADD CONSTRAINT "FeaturedAgent_agentId_fkey" FOREIGN KEY ("agentId") REFERENCES "Agents"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
