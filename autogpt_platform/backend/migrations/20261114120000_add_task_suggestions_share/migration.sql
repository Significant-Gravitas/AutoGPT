-- Tasks Otto drafts for the user to start or dismiss.

-- AlterTable
ALTER TABLE "Task" ADD COLUMN     "isSuggested" BOOLEAN NOT NULL DEFAULT false,
ADD COLUMN     "suggestedReason" TEXT,
ADD COLUMN     "dismissedAt" TIMESTAMP(3);
