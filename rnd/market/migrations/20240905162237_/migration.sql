-- CreateEnum
CREATE TYPE "InstallationLocation" AS ENUM ('LOCAL', 'CLOUD');

-- CreateTable
CREATE TABLE "InstallTracker" (
    "id" UUID NOT NULL DEFAULT gen_random_uuid(),
    "marketplaceAgentId" UUID NOT NULL,
    "installedAgentId" UUID NOT NULL,
    "installationLocation" "InstallationLocation" NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "InstallTracker_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "InstallTracker_marketplaceAgentId_installedAgentId_key" ON "InstallTracker"("marketplaceAgentId", "installedAgentId");

-- AddForeignKey
ALTER TABLE "InstallTracker" ADD CONSTRAINT "InstallTracker_marketplaceAgentId_fkey" FOREIGN KEY ("marketplaceAgentId") REFERENCES "Agents"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
