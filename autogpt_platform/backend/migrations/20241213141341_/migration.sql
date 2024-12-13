-- AlterTable
ALTER TABLE "CreditTransaction" RENAME CONSTRAINT "UserBlockCredit_pkey" TO "CreditTransaction_pkey";

-- RenameForeignKey
ALTER TABLE "CreditTransaction" RENAME CONSTRAINT "UserBlockCredit_blockId_fkey" TO "CreditTransaction_blockId_fkey";

-- RenameForeignKey
ALTER TABLE "CreditTransaction" RENAME CONSTRAINT "UserBlockCredit_userId_fkey" TO "CreditTransaction_userId_fkey";

-- RenameIndex
ALTER INDEX "UserBlockCredit_userId_createdAt_idx" RENAME TO "CreditTransaction_userId_createdAt_idx";
