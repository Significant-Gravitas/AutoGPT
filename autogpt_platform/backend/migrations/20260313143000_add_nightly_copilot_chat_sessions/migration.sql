-- CreateEnum
CREATE TYPE "ChatSessionStartType" AS ENUM(
  'MANUAL',
  'AUTOPILOT_NIGHTLY',
  'AUTOPILOT_CALLBACK',
  'AUTOPILOT_INVITE_CTA'
);

-- AlterTable
ALTER TABLE "ChatSession"
ADD COLUMN "startType" "ChatSessionStartType" NOT NULL DEFAULT 'MANUAL',
ADD COLUMN "executionTag" TEXT,
ADD COLUMN "sessionConfig" JSONB NOT NULL DEFAULT '{}',
ADD COLUMN "completionReport" JSONB,
ADD COLUMN "completionReportRepairCount" INTEGER NOT NULL DEFAULT 0,
ADD COLUMN "completionReportRepairQueuedAt" TIMESTAMP(3),
ADD COLUMN "completedAt" TIMESTAMP(3),
ADD COLUMN "notificationEmailSentAt" TIMESTAMP(3),
ADD COLUMN "notificationEmailSkippedAt" TIMESTAMP(3);

COMMENT ON COLUMN "ChatSession"."sessionConfig" IS 'Validated by backend.copilot.session_types.ChatSessionConfig';
COMMENT ON COLUMN "ChatSession"."completionReport" IS 'Validated by backend.copilot.session_types.StoredCompletionReport';

-- CreateTable
CREATE TABLE "ChatSessionCallbackToken"(
  "id" TEXT NOT NULL,
  "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "userId" TEXT NOT NULL,
  "sourceSessionId" TEXT,
  "callbackSessionMessage" TEXT NOT NULL,
  "expiresAt" TIMESTAMP(3) NOT NULL,
  "consumedAt" TIMESTAMP(3),
  "consumedSessionId" TEXT,
  CONSTRAINT "ChatSessionCallbackToken_pkey" PRIMARY KEY("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "ChatSession_userId_executionTag_key"
ON "ChatSession"("userId",
  "executionTag");

-- CreateIndex
CREATE INDEX "ChatSession_userId_startType_updatedAt_idx"
ON "ChatSession"("userId",
  "startType",
  "updatedAt");

-- CreateIndex
CREATE INDEX "ChatSessionCallbackToken_userId_expiresAt_idx"
ON "ChatSessionCallbackToken"("userId",
  "expiresAt");

-- CreateIndex
CREATE INDEX "ChatSessionCallbackToken_consumedSessionId_idx"
ON "ChatSessionCallbackToken"("consumedSessionId");

-- AddForeignKey
ALTER TABLE "ChatSessionCallbackToken" ADD CONSTRAINT "ChatSessionCallbackToken_userId_fkey" FOREIGN KEY("userId") REFERENCES "User"("id")
ON DELETE CASCADE
ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ChatSessionCallbackToken" ADD CONSTRAINT "ChatSessionCallbackToken_sourceSessionId_fkey" FOREIGN KEY("sourceSessionId") REFERENCES "ChatSession"("id")
ON DELETE 
CASCADE
ON UPDATE CASCADE;
