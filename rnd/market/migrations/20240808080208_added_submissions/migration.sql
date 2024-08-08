-- CreateEnum
CREATE TYPE "SubmissionStatus" AS ENUM ('PENDING', 'APPROVED', 'REJECTED');

-- AlterTable
ALTER TABLE "Agents" ADD COLUMN     "submissionDate" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
ADD COLUMN     "submissionReviewComments" TEXT,
ADD COLUMN     "submissionReviewDate" TIMESTAMP(3),
ADD COLUMN     "submissionStatus" "SubmissionStatus" NOT NULL DEFAULT 'PENDING';
