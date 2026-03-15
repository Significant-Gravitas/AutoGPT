-- CreateEnum
CREATE TYPE "APIKeyPermission" AS ENUM ('EXECUTE_GRAPH', 'READ_GRAPH', 'EXECUTE_BLOCK', 'READ_BLOCK');

-- CreateEnum
CREATE TYPE "APIKeyStatus" AS ENUM ('ACTIVE', 'REVOKED', 'SUSPENDED');

-- CreateTable
CREATE TABLE "APIKey" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "prefix" TEXT NOT NULL,
    "postfix" TEXT NOT NULL,
    "key" TEXT NOT NULL,
    "status" "APIKeyStatus" NOT NULL DEFAULT 'ACTIVE',
    "permissions" "APIKeyPermission"[],
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "lastUsedAt" TIMESTAMP(3),
    "revokedAt" TIMESTAMP(3),
    "description" TEXT,
    "userId" TEXT NOT NULL,

    CONSTRAINT "APIKey_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "APIKey_key_key" ON "APIKey"("key");

-- CreateIndex
CREATE INDEX "APIKey_key_idx" ON "APIKey"("key");

-- CreateIndex
CREATE INDEX "APIKey_prefix_idx" ON "APIKey"("prefix");

-- CreateIndex
CREATE INDEX "APIKey_userId_idx" ON "APIKey"("userId");

-- CreateIndex
CREATE INDEX "APIKey_status_idx" ON "APIKey"("status");

-- CreateIndex
CREATE INDEX "APIKey_userId_status_idx" ON "APIKey"("userId", "status");

-- AddForeignKey
ALTER TABLE "APIKey" ADD CONSTRAINT "APIKey_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
