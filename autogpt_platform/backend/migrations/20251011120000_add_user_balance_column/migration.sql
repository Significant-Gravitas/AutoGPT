-- Create UserBalance table for atomic credit operations
-- This replaces the need for User.balance column and provides better separation of concerns
-- UserBalance records are automatically created by the application when users interact with the credit system

-- CreateTable (only if it doesn't exist)
CREATE TABLE IF NOT EXISTS "UserBalance" (
    "userId" TEXT NOT NULL,
    "balance" INTEGER NOT NULL DEFAULT 0,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "UserBalance_pkey" PRIMARY KEY ("userId"),
    CONSTRAINT "UserBalance_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateIndex (only if it doesn't exist)
CREATE INDEX IF NOT EXISTS "UserBalance_userId_idx" ON "UserBalance"("userId");
