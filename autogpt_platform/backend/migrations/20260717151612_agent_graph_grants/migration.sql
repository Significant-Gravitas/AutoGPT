-- CreateEnum
CREATE TYPE "GrantPrincipalType" AS ENUM ('TEAM', 'USER');

-- CreateEnum
CREATE TYPE "GrantCapability" AS ENUM ('VIEW', 'EXECUTE');

-- CreateEnum
CREATE TYPE "GrantCredentialMode" AS ENUM ('CONSUMER', 'OWNER');

-- CreateTable
CREATE TABLE "AgentGraphGrant" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "agentGraphId" TEXT NOT NULL,
    "agentGraphVersion" INTEGER NOT NULL,
    "followLatest" BOOLEAN NOT NULL DEFAULT false,
    "principalType" "GrantPrincipalType" NOT NULL,
    "principalId" TEXT NOT NULL,
    "capability" "GrantCapability" NOT NULL DEFAULT 'EXECUTE',
    "credentialMode" "GrantCredentialMode" NOT NULL DEFAULT 'CONSUMER',
    "organizationId" TEXT NOT NULL,
    "createdByUserId" TEXT NOT NULL,

    CONSTRAINT "AgentGraphGrant_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "AgentGraphGrant_principalType_principalId_idx" ON "AgentGraphGrant"("principalType", "principalId");

-- CreateIndex
CREATE INDEX "AgentGraphGrant_organizationId_idx" ON "AgentGraphGrant"("organizationId");

-- CreateIndex
CREATE INDEX "AgentGraphGrant_agentGraphId_agentGraphVersion_idx" ON "AgentGraphGrant"("agentGraphId", "agentGraphVersion");

-- CreateIndex
CREATE UNIQUE INDEX "AgentGraphGrant_agentGraphId_principalType_principalId_key" ON "AgentGraphGrant"("agentGraphId", "principalType", "principalId");

-- AddForeignKey
ALTER TABLE "AgentGraphGrant" ADD CONSTRAINT "AgentGraphGrant_agentGraphId_agentGraphVersion_fkey" FOREIGN KEY ("agentGraphId", "agentGraphVersion") REFERENCES "AgentGraph"("id", "version") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AgentGraphGrant" ADD CONSTRAINT "AgentGraphGrant_organizationId_fkey" FOREIGN KEY ("organizationId") REFERENCES "Organization"("id") ON DELETE CASCADE ON UPDATE CASCADE;

