-- DropForeignKey
ALTER TABLE "AnalyticsTracker" DROP CONSTRAINT "AnalyticsTracker_agentId_fkey";

-- DropForeignKey
ALTER TABLE "FeaturedAgent" DROP CONSTRAINT "FeaturedAgent_agentId_fkey";

-- DropForeignKey
ALTER TABLE "InstallTracker" DROP CONSTRAINT "InstallTracker_marketplaceAgentId_fkey";

-- DropIndex
DROP INDEX "AnalyticsTracker_agentId_key";

-- AddForeignKey
ALTER TABLE "AnalyticsTracker" ADD CONSTRAINT "AnalyticsTracker_agentId_fkey" FOREIGN KEY ("agentId") REFERENCES "Agents"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "InstallTracker" ADD CONSTRAINT "InstallTracker_marketplaceAgentId_fkey" FOREIGN KEY ("marketplaceAgentId") REFERENCES "Agents"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "FeaturedAgent" ADD CONSTRAINT "FeaturedAgent_agentId_fkey" FOREIGN KEY ("agentId") REFERENCES "Agents"("id") ON DELETE CASCADE ON UPDATE CASCADE;
