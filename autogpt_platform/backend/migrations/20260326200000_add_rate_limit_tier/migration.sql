-- CreateEnum
CREATE TYPE "SubscriptionTier" AS ENUM ('FREE', 'PRO', 'BUSINESS', 'ENTERPRISE');

-- AlterTable: add subscriptionTier column with default PRO (beta testing)
ALTER TABLE "User" ADD COLUMN "subscriptionTier" "SubscriptionTier" NOT NULL DEFAULT 'PRO';
