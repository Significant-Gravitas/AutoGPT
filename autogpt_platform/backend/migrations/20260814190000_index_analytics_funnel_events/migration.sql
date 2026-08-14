DROP INDEX IF EXISTS "AnalyticsDetails_userId_type_idx";

CREATE INDEX "AnalyticsDetails_userId_type_dataIndex_idx"
    ON "AnalyticsDetails"("userId", "type", "dataIndex");
