-- DropForeignKey
ALTER TABLE "AnalyticsTracker" DROP CONSTRAINT "AnalyticsTracker_agentId_fkey";

-- DropIndex
DROP INDEX "AnalyticsTracker_agentId_key";

-- AddForeignKey
ALTER TABLE "AnalyticsTracker" ADD CONSTRAINT "AnalyticsTracker_agentId_fkey" FOREIGN KEY ("agentId") REFERENCES "Agents"("id") ON DELETE CASCADE ON UPDATE CASCADE;
