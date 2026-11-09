-- In-app feedback prompts: one row per showing, so each prompt's limit holds
-- across tabs and devices. Also the picked option on a work check or mistake
-- rating of a chat reply.

-- Nullable with no default: a catalog-only change, no table rewrite.
-- AlterTable
ALTER TABLE "ChatMessageFeedback" ADD COLUMN     "reason" TEXT;

-- CreateTable
CREATE TABLE "FeedbackPrompt" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "kind" TEXT NOT NULL,
    "sessionId" TEXT,
    "messageId" TEXT,
    "shownAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "answeredAt" TIMESTAMP(3),
    "dismissedAt" TIMESTAMP(3),
    "reason" TEXT,
    "note" TEXT,
    "score" INTEGER,
    "offerTaken" BOOLEAN,

    CONSTRAINT "FeedbackPrompt_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "FeedbackPrompt_userId_kind_shownAt_idx" ON "FeedbackPrompt"("userId", "kind", "shownAt");

-- AddForeignKey
ALTER TABLE "FeedbackPrompt" ADD CONSTRAINT "FeedbackPrompt_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
