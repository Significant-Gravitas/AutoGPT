-- Add balance column to User table for atomic credit operations
-- This replaces the need for advisory locks and expensive transaction aggregations

-- Add balance column with default 0 for new users
ALTER TABLE "User" ADD COLUMN balance INT DEFAULT 0;

-- Backfill ALL user balances from transaction history
-- Users with transactions: use their latest runningBalance
-- Users without transactions: set to 0
UPDATE "User" 
SET balance = COALESCE(latest_balances.latest_running_balance, 0)
FROM (
    SELECT 
        u.id as user_id,
        latest_tx.latest_running_balance
    FROM "User" u
    LEFT JOIN (
        SELECT DISTINCT ON (ct."userId") 
            ct."userId" as user_id,
            ct."runningBalance" as latest_running_balance
        FROM "CreditTransaction" ct
        WHERE ct."isActive" = true 
          AND ct."runningBalance" IS NOT NULL
        ORDER BY ct."userId", ct."createdAt" DESC
    ) latest_tx ON u.id = latest_tx.user_id
) latest_balances
WHERE "User".id = latest_balances.user_id;