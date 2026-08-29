-- AlterTable
ALTER TABLE "User" ADD COLUMN "stripeCustomerId" TEXT;

-- AlterEnum
ALTER TYPE "UserBlockCreditType" RENAME TO "CreditTransactionType";

-- AlterTable
ALTER TABLE "UserBlockCredit" RENAME TO "CreditTransaction";
