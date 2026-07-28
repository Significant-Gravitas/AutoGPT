-- CreateEnum
CREATE TYPE "PresetDeactivationReason" AS ENUM ('PAYMENT_LAPSED');

-- AlterEnum
ALTER TYPE "NotificationType" ADD VALUE IF NOT EXISTS 'AUTOMATIONS_PAUSED';
ALTER TYPE "NotificationType" ADD VALUE IF NOT EXISTS 'AUTOMATIONS_RESUMED';

-- AlterTable
ALTER TABLE "AgentPreset" ADD COLUMN "deactivationReason" "PresetDeactivationReason";

-- AlterTable
ALTER TABLE "User" ADD COLUMN "notifyOnAutomationsPaused" BOOLEAN NOT NULL DEFAULT true;
