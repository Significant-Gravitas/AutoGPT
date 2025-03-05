-- Create UserOnboarding table
CREATE TABLE "UserOnboarding" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "step" INTEGER NOT NULL DEFAULT 0,
    "usageReason" TEXT,
    "integrations" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "otherIntegrations" TEXT,
    "selectedAgentCreator" TEXT,
    "selectedAgentSlug" TEXT,
    "agentInput" JSONB,
    "isCompleted" BOOLEAN NOT NULL DEFAULT false,
    "userId" TEXT NOT NULL,

    CONSTRAINT "UserOnboarding_pkey" PRIMARY KEY ("id")
);

-- Create unique constraint on userId
ALTER TABLE "UserOnboarding" ADD CONSTRAINT "UserOnboarding_userId_key" UNIQUE ("userId");

-- Create index on userId
CREATE INDEX "UserOnboarding_userId_idx" ON "UserOnboarding"("userId");

-- Add foreign key constraint
ALTER TABLE "UserOnboarding" ADD CONSTRAINT "UserOnboarding_userId_fkey" 
FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
