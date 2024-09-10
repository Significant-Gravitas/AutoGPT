-- CreateEnum
CREATE TYPE "AnalyticsType" AS ENUM ('CREATE_USER', 'TUTORIAL_STEP', 'WEB_PAGE', 'AGENT_GRAPH_EXECUTION', 'AGENT_NODE_EXECUTION');

-- CreateEnum
CREATE TYPE "AnalyticsMetric" AS ENUM ('PAGE_VIEW', 'TUTORIAL_STEP_COMPLETION', 'AGENT_GRAPH_EXECUTION', 'AGENT_NODE_EXECUTION');

-- CreateEnum
CREATE TYPE "AggregationType" AS ENUM ('COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'NO_AGGREGATION');

-- CreateTable
CREATE TABLE "AnalyticsDetails" (
    "id" TEXT NOT NULL DEFAULT gen_random_uuid(),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT NOT NULL,
    "type" "AnalyticsType" NOT NULL,
    "data" JSONB,
    "dataIndex" INTEGER,

    CONSTRAINT "AnalyticsDetails_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "AnalyticsMetrics" (
    "id" TEXT NOT NULL DEFAULT gen_random_uuid(),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "analyticMetric" "AnalyticsMetric" NOT NULL,
    "value" DOUBLE PRECISION NOT NULL,
    "dataString" TEXT,
    "aggregationType" "AggregationType" NOT NULL DEFAULT 'NO_AGGREGATION',
    "userId" TEXT NOT NULL,

    CONSTRAINT "AnalyticsMetrics_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "analyticsDetails" ON "AnalyticsDetails"("userId", "type");

-- CreateIndex
CREATE INDEX "AnalyticsDetails_type_idx" ON "AnalyticsDetails"("type");

-- CreateIndex
CREATE INDEX "analytics_metric_index" ON "AnalyticsMetrics"("analyticMetric", "userId", "dataString", "aggregationType");

-- CreateIndex
CREATE UNIQUE INDEX "AnalyticsMetrics_analyticMetric_userId_dataString_aggregati_key" ON "AnalyticsMetrics"("analyticMetric", "userId", "dataString", "aggregationType");

-- AddForeignKey
ALTER TABLE "AnalyticsDetails" ADD CONSTRAINT "AnalyticsDetails_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AnalyticsMetrics" ADD CONSTRAINT "AnalyticsMetrics_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
