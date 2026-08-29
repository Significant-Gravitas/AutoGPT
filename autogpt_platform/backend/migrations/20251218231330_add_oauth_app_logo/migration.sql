-- AlterEnum
ALTER TYPE "APIKeyPermission" ADD VALUE 'IDENTITY';

-- AlterTable
ALTER TABLE "OAuthApplication" ADD COLUMN     "logoUrl" TEXT;
