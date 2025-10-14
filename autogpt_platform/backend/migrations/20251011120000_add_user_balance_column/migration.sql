-- Create UserBalance table for atomic credit operations
-- This replaces the need for User.balance column and provides better separation of concerns

-- Create UserBalance table
CREATE TABLE "UserBalance" (
  "id" TEXT NOT NULL,
  "userId" TEXT NOT NULL,
  "balance" INTEGER NOT NULL DEFAULT 0,
  "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  
  CONSTRAINT "UserBalance_pkey" PRIMARY KEY ("id"),
  CONSTRAINT "UserBalance_userId_key" UNIQUE ("userId"),
  CONSTRAINT "UserBalance_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- Create index for efficient lookups
CREATE INDEX "UserBalance_userId_idx" ON "UserBalance"("userId");

-- Migrate existing user balances from transaction history
-- Users with transactions: use their latest runningBalance
-- Users without transactions: create with balance 0
INSERT INTO "UserBalance" ("id", "userId", "balance", "updatedAt")
SELECT 
    gen_random_uuid()::text as id,
    u.id as "userId",
    COALESCE(latest_balances.latest_running_balance, 0) as balance,
    COALESCE(latest_balances.last_transaction_time, u."updatedAt") as "updatedAt"
FROM "User" u
LEFT JOIN (
    SELECT DISTINCT ON (ct."userId") 
        ct."userId" as user_id,
        ct."runningBalance" as latest_running_balance,
        ct."createdAt" as last_transaction_time
    FROM "CreditTransaction" ct
    WHERE ct."isActive" = true 
      AND ct."runningBalance" IS NOT NULL
    ORDER BY ct."userId", ct."createdAt" DESC
) latest_balances ON u.id = latest_balances.user_id;