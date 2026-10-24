-- CreateEnum
CREATE TYPE "WorkloadIdentityStatus" AS ENUM ('ACTIVE', 'DISABLED', 'REVOKED');

-- CreateTable
CREATE TABLE "WorkloadIdentity" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "issuer" TEXT NOT NULL,
    "audience" TEXT NOT NULL,
    "claimConditions" JSONB NOT NULL,
    "permissions" "APIKeyPermission"[],
    "tokenTtlSeconds" INTEGER NOT NULL DEFAULT 900,
    "status" "WorkloadIdentityStatus" NOT NULL DEFAULT 'ACTIVE',
    "lastUsedAt" TIMESTAMP(3),
    "revokedAt" TIMESTAMP(3),
    "userId" TEXT NOT NULL,
    "organizationId" TEXT,

    CONSTRAINT "WorkloadIdentity_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "WorkloadAccessToken" (
    "id" TEXT NOT NULL,
    "token" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "expiresAt" TIMESTAMP(3) NOT NULL,
    "workloadIdentityId" TEXT NOT NULL,
    "scopes" "APIKeyPermission"[],
    "subject" TEXT NOT NULL,
    "subjectJti" TEXT,
    "revokedAt" TIMESTAMP(3),

    CONSTRAINT "WorkloadAccessToken_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "WorkloadIdentity_userId_status_idx" ON "WorkloadIdentity"("userId", "status");

-- CreateIndex
CREATE INDEX "WorkloadIdentity_organizationId_idx" ON "WorkloadIdentity"("organizationId");

-- CreateIndex
CREATE UNIQUE INDEX "WorkloadAccessToken_token_key" ON "WorkloadAccessToken"("token");

-- CreateIndex
CREATE INDEX "WorkloadAccessToken_workloadIdentityId_idx" ON "WorkloadAccessToken"("workloadIdentityId");

-- CreateIndex
CREATE INDEX "WorkloadAccessToken_expiresAt_idx" ON "WorkloadAccessToken"("expiresAt");

-- AddForeignKey
ALTER TABLE "WorkloadIdentity" ADD CONSTRAINT "WorkloadIdentity_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "WorkloadAccessToken" ADD CONSTRAINT "WorkloadAccessToken_workloadIdentityId_fkey" FOREIGN KEY ("workloadIdentityId") REFERENCES "WorkloadIdentity"("id") ON DELETE CASCADE ON UPDATE CASCADE;

