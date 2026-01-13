-- DropIndex
DROP INDEX "StoreListingVersion_storeListingId_version_key";

-- CreateTable
CREATE TABLE "UserBusinessUnderstanding" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT NOT NULL,
    "usersName" TEXT,
    "jobTitle" TEXT,
    "businessName" TEXT,
    "industry" TEXT,
    "businessSize" TEXT,
    "userRole" TEXT,
    "keyWorkflows" JSONB,
    "dailyActivities" JSONB,
    "painPoints" JSONB,
    "bottlenecks" JSONB,
    "manualTasks" JSONB,
    "automationGoals" JSONB,
    "currentSoftware" JSONB,
    "existingAutomation" JSONB,
    "additionalNotes" TEXT,

    CONSTRAINT "UserBusinessUnderstanding_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ChatSession" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT,
    "title" TEXT,
    "credentials" JSONB NOT NULL DEFAULT '{}',
    "successfulAgentRuns" JSONB NOT NULL DEFAULT '{}',
    "successfulAgentSchedules" JSONB NOT NULL DEFAULT '{}',
    "totalPromptTokens" INTEGER NOT NULL DEFAULT 0,
    "totalCompletionTokens" INTEGER NOT NULL DEFAULT 0,

    CONSTRAINT "ChatSession_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ChatMessage" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "sessionId" TEXT NOT NULL,
    "role" TEXT NOT NULL,
    "content" TEXT,
    "name" TEXT,
    "toolCallId" TEXT,
    "refusal" TEXT,
    "toolCalls" JSONB,
    "functionCall" JSONB,
    "sequence" INTEGER NOT NULL,

    CONSTRAINT "ChatMessage_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "UserBusinessUnderstanding_userId_key" ON "UserBusinessUnderstanding"("userId");

-- CreateIndex
CREATE INDEX "UserBusinessUnderstanding_userId_idx" ON "UserBusinessUnderstanding"("userId");

-- CreateIndex
CREATE INDEX "ChatSession_userId_updatedAt_idx" ON "ChatSession"("userId", "updatedAt");

-- CreateIndex
CREATE INDEX "ChatMessage_sessionId_sequence_idx" ON "ChatMessage"("sessionId", "sequence");

-- CreateIndex
CREATE UNIQUE INDEX "ChatMessage_sessionId_sequence_key" ON "ChatMessage"("sessionId", "sequence");

-- AddForeignKey
ALTER TABLE "UserBusinessUnderstanding" ADD CONSTRAINT "UserBusinessUnderstanding_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ChatMessage" ADD CONSTRAINT "ChatMessage_sessionId_fkey" FOREIGN KEY ("sessionId") REFERENCES "ChatSession"("id") ON DELETE CASCADE ON UPDATE CASCADE;
