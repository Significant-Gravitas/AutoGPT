-- AlterEnum
-- This migration adds more than one value to an enum.
-- With PostgreSQL versions 11 and earlier, this is not possible
-- in a single migration. This can be worked around by creating
-- multiple migrations, each migration adding only one value to
-- the enum.


ALTER TYPE "NotificationType" ADD VALUE 'AGENT_APPROVED';
ALTER TYPE "NotificationType" ADD VALUE 'AGENT_REJECTED';

-- AlterTable
ALTER TABLE "User" ADD COLUMN     "notifyOnAgentApproved" BOOLEAN NOT NULL DEFAULT true,
ADD COLUMN     "notifyOnAgentRejected" BOOLEAN NOT NULL DEFAULT true;
