-- Thumbs up/down and copy ratings on AutoPilot replies, plus the Langfuse
-- trace id of the SDK turn that wrote each assistant row so a rating can be
-- scored against that trace.

-- Nullable with no default: a catalog-only change, no table rewrite.
-- AlterTable
ALTER TABLE "ChatMessage" ADD COLUMN     "langfuseTraceId" TEXT;

-- CreateTable
CREATE TABLE "ChatMessageFeedback" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT NOT NULL,
    "messageId" TEXT NOT NULL,
    "sessionId" TEXT NOT NULL,
    "scoreName" TEXT NOT NULL,
    "scoreValue" INTEGER NOT NULL,
    "comment" TEXT,
    "langfuseTraceId" TEXT,

    CONSTRAINT "ChatMessageFeedback_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "ChatMessageFeedback_userId_idx" ON "ChatMessageFeedback"("userId");

-- CreateIndex
CREATE UNIQUE INDEX "ChatMessageFeedback_messageId_userId_scoreName_key" ON "ChatMessageFeedback"("messageId", "userId", "scoreName");

-- AddForeignKey
ALTER TABLE "ChatMessageFeedback" ADD CONSTRAINT "ChatMessageFeedback_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ChatMessageFeedback" ADD CONSTRAINT "ChatMessageFeedback_messageId_fkey" FOREIGN KEY ("messageId") REFERENCES "ChatMessage"("id") ON DELETE CASCADE ON UPDATE CASCADE;
