/*
  Warnings:

  - You are about to drop the column `aggregationCounter` on the `AnalyticsMetrics` table. All the data in the column will be lost.
  - You are about to drop the column `aggregationType` on the `AnalyticsMetrics` table. All the data in the column will be lost.
  - Changed the type of `type` on the `AnalyticsDetails` table. No cast exists, the column would be dropped and recreated, which cannot be done if there is data, since the column is required.
  - Changed the type of `analyticMetric` on the `AnalyticsMetrics` table. No cast exists, the column would be dropped and recreated, which cannot be done if there is data, since the column is required.

*/
-- DropIndex
DROP INDEX "AnalyticsMetrics_analyticMetric_userId_dataString_aggregati_key";

-- DropIndex
DROP INDEX "analytics_metric_index";

-- AlterTable
ALTER TABLE "AnalyticsDetails" DROP COLUMN "type",
ADD COLUMN     "type" TEXT NOT NULL;

-- AlterTable
ALTER TABLE "AnalyticsMetrics" DROP COLUMN "aggregationCounter",
DROP COLUMN "aggregationType",
DROP COLUMN "analyticMetric",
ADD COLUMN     "analyticMetric" TEXT NOT NULL;

-- DropEnum
DROP TYPE "AggregationType";

-- DropEnum
DROP TYPE "AnalyticsMetric";

-- DropEnum
DROP TYPE "AnalyticsType";

-- CreateIndex
CREATE INDEX "analyticsDetails" ON "AnalyticsDetails"("userId", "type");

-- CreateIndex
CREATE INDEX "AnalyticsDetails_type_idx" ON "AnalyticsDetails"("type");

-- CreateIndex
CREATE INDEX "analytics_metric_index" ON "AnalyticsMetrics"("analyticMetric", "userId", "dataString");
