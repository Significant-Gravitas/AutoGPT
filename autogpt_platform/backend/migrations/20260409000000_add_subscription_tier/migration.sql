-- SubscriptionTier enum and User.subscriptionTier column already created by
-- 20260326200000_add_rate_limit_tier migration. Only add SUBSCRIPTION transaction type.

-- AlterEnum
ALTER TYPE "CreditTransactionType" ADD VALUE IF NOT EXISTS 'SUBSCRIPTION';
