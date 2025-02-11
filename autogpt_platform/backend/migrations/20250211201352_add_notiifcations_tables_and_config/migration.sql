/*
  Warnings:

  - A unique constraint covering the columns `[userNotificationPreferenceId]` on the table `User` will be added. If there are existing duplicate values, this will fail.

*/
-- CreateEnum
CREATE TYPE "NotificationType" AS ENUM ('AGENT_RUN', 'ZERO_BALANCE', 'LOW_BALANCE', 'BLOCK_EXECUTION_FAILED', 'CONTINUOUS_AGENT_ERROR', 'DAILY_SUMMARY', 'WEEKLY_SUMMARY', 'MONTHLY_SUMMARY');

-- AlterTable
ALTER TABLE "User" ADD COLUMN     "userNotificationPreferenceId" TEXT;

-- CreateTable
CREATE TABLE "NotificationEvent" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userNotificationBatchId" TEXT,
    "type" "NotificationType" NOT NULL,
    "data" JSONB NOT NULL,

    CONSTRAINT "NotificationEvent_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "UserNotificationBatch" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT NOT NULL,
    "type" "NotificationType" NOT NULL,

    CONSTRAINT "UserNotificationBatch_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "UserNotificationPreference" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT NOT NULL,
    "maxEmailsPerDay" INTEGER NOT NULL DEFAULT 3,
    "notifyOnAgentRun" BOOLEAN NOT NULL DEFAULT true,
    "notifyOnZeroBalance" BOOLEAN NOT NULL DEFAULT true,
    "notifyOnLowBalance" BOOLEAN NOT NULL DEFAULT true,
    "notifyOnBlockExecutionFailed" BOOLEAN NOT NULL DEFAULT true,
    "notifyOnContinuousAgentError" BOOLEAN NOT NULL DEFAULT true,
    "notifyOnDailySummary" BOOLEAN NOT NULL DEFAULT true,
    "notifyOnWeeklySummary" BOOLEAN NOT NULL DEFAULT true,
    "notifyOnMonthlySummary" BOOLEAN NOT NULL DEFAULT true,

    CONSTRAINT "UserNotificationPreference_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "UserNotificationBatch_userId_type_key" ON "UserNotificationBatch"("userId", "type");

-- CreateIndex
CREATE UNIQUE INDEX "UserNotificationPreference_userId_key" ON "UserNotificationPreference"("userId");

-- CreateIndex
CREATE UNIQUE INDEX "User_userNotificationPreferenceId_key" ON "User"("userNotificationPreferenceId");

-- AddForeignKey
ALTER TABLE "User" ADD CONSTRAINT "User_userNotificationPreferenceId_fkey" FOREIGN KEY ("userNotificationPreferenceId") REFERENCES "UserNotificationPreference"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "NotificationEvent" ADD CONSTRAINT "NotificationEvent_userNotificationBatchId_fkey" FOREIGN KEY ("userNotificationBatchId") REFERENCES "UserNotificationBatch"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "UserNotificationBatch" ADD CONSTRAINT "UserNotificationBatch_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
