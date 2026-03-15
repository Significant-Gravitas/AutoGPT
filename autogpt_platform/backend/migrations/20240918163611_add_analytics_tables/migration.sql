-- CreateTable
CREATE TABLE "AnalyticsDetails" (
    "id" TEXT NOT NULL DEFAULT gen_random_uuid(),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "data" JSONB,
    "dataIndex" TEXT,

    CONSTRAINT "AnalyticsDetails_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "AnalyticsMetrics" (
    "id" TEXT NOT NULL DEFAULT gen_random_uuid(),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "analyticMetric" TEXT NOT NULL,
    "value" DOUBLE PRECISION NOT NULL,
    "dataString" TEXT,
    "userId" TEXT NOT NULL,

    CONSTRAINT "AnalyticsMetrics_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "analyticsDetails" ON "AnalyticsDetails"("userId", "type");

-- CreateIndex
CREATE INDEX "AnalyticsDetails_type_idx" ON "AnalyticsDetails"("type");

-- AddForeignKey
ALTER TABLE "AnalyticsDetails" ADD CONSTRAINT "AnalyticsDetails_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AnalyticsMetrics" ADD CONSTRAINT "AnalyticsMetrics_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
