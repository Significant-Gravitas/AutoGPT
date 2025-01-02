/*
  Warnings:

  - A unique constraint covering the columns `[agentId]` on the table `AnalyticsTracker` will be added. If there are existing duplicate values, this will fail.

*/
-- AlterTable
ALTER TABLE "AnalyticsTracker" ADD CONSTRAINT "AnalyticsTracker_pkey" PRIMARY KEY ("id");

-- CreateIndex
CREATE UNIQUE INDEX "AnalyticsTracker_agentId_key" ON "AnalyticsTracker"("agentId");
