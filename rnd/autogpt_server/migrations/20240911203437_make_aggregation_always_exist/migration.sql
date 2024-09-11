/*
  Warnings:

  - Made the column `aggregationCounter` on table `AnalyticsMetrics` required. This step will fail if there are existing NULL values in that column.

*/
-- AlterTable
ALTER TABLE "AnalyticsMetrics" ALTER COLUMN "aggregationCounter" SET NOT NULL;
