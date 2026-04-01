-- CreateEnum
CREATE TYPE "WorkspaceJoinPolicy" AS ENUM ('OPEN', 'PRIVATE');

-- CreateEnum
CREATE TYPE "ResourceVisibility" AS ENUM ('PRIVATE', 'WORKSPACE', 'ORG');

-- CreateEnum
CREATE TYPE "CredentialScope" AS ENUM ('USER', 'WORKSPACE', 'ORG');

-- CreateEnum
CREATE TYPE "OrgAliasType" AS ENUM ('MIGRATION', 'RENAME', 'MANUAL');

-- CreateEnum
CREATE TYPE "OrgMemberStatus" AS ENUM ('INVITED', 'ACTIVE', 'SUSPENDED', 'REMOVED');

-- CreateEnum
CREATE TYPE "SeatType" AS ENUM ('FREE', 'PAID');

-- CreateEnum
CREATE TYPE "SeatStatus" AS ENUM ('ACTIVE', 'INACTIVE', 'PENDING');

-- CreateEnum
CREATE TYPE "TransferStatus" AS ENUM ('PENDING', 'SOURCE_APPROVED', 'TARGET_APPROVED', 'COMPLETED', 'REJECTED', 'CANCELLED');

-- CreateEnum
CREATE TYPE "CredentialOwnerType" AS ENUM ('USER', 'WORKSPACE', 'ORG');

-- AlterTable
ALTER TABLE "BuilderSearchHistory" ADD COLUMN     "organizationId" TEXT;

-- AlterTable
ALTER TABLE "ChatSession" ADD COLUMN     "orgWorkspaceId" TEXT,
ADD COLUMN     "organizationId" TEXT,
ADD COLUMN     "visibility" "ResourceVisibility" NOT NULL DEFAULT 'PRIVATE';

-- AlterTable
ALTER TABLE "AgentGraph" ADD COLUMN     "orgWorkspaceId" TEXT,
ADD COLUMN     "organizationId" TEXT,
ADD COLUMN     "visibility" "ResourceVisibility" NOT NULL DEFAULT 'PRIVATE';

-- AlterTable
ALTER TABLE "AgentPreset" ADD COLUMN     "orgWorkspaceId" TEXT,
ADD COLUMN     "organizationId" TEXT,
ADD COLUMN     "visibility" "ResourceVisibility" NOT NULL DEFAULT 'PRIVATE';

-- AlterTable
ALTER TABLE "LibraryAgent" ADD COLUMN     "orgWorkspaceId" TEXT,
ADD COLUMN     "organizationId" TEXT,
ADD COLUMN     "visibility" "ResourceVisibility" NOT NULL DEFAULT 'PRIVATE';

-- AlterTable
ALTER TABLE "LibraryFolder" ADD COLUMN     "orgWorkspaceId" TEXT,
ADD COLUMN     "organizationId" TEXT,
ADD COLUMN     "visibility" "ResourceVisibility" NOT NULL DEFAULT 'PRIVATE';

-- AlterTable
ALTER TABLE "AgentGraphExecution" ADD COLUMN     "orgWorkspaceId" TEXT,
ADD COLUMN     "organizationId" TEXT,
ADD COLUMN     "visibility" "ResourceVisibility" NOT NULL DEFAULT 'PRIVATE';

-- AlterTable
ALTER TABLE "PendingHumanReview" ADD COLUMN     "orgWorkspaceId" TEXT,
ADD COLUMN     "organizationId" TEXT;

-- AlterTable
ALTER TABLE "IntegrationWebhook" ADD COLUMN     "orgWorkspaceId" TEXT,
ADD COLUMN     "organizationId" TEXT,
ADD COLUMN     "visibility" "ResourceVisibility" NOT NULL DEFAULT 'PRIVATE';

-- AlterTable
ALTER TABLE "StoreListing" ADD COLUMN     "owningOrgId" TEXT;

-- AlterTable
ALTER TABLE "StoreListingVersion" ADD COLUMN     "organizationId" TEXT;

-- AlterTable
ALTER TABLE "APIKey" ADD COLUMN     "orgWorkspaceId" TEXT,
ADD COLUMN     "organizationId" TEXT,
ADD COLUMN     "ownerType" "CredentialOwnerType",
ADD COLUMN     "workspaceIdRestriction" TEXT;

-- AlterTable
ALTER TABLE "OAuthApplication" ADD COLUMN     "organizationId" TEXT,
ADD COLUMN     "ownerType" "CredentialOwnerType",
ADD COLUMN     "workspaceIdRestriction" TEXT;

-- CreateTable
CREATE TABLE "Organization" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "name" TEXT NOT NULL,
    "slug" TEXT NOT NULL,
    "avatarUrl" TEXT,
    "description" TEXT,
    "isPersonal" BOOLEAN NOT NULL DEFAULT false,
    "settings" JSONB NOT NULL DEFAULT '{}',
    "stripeCustomerId" TEXT,
    "stripeSubscriptionId" TEXT,
    "topUpConfig" JSONB,
    "archivedAt" TIMESTAMP(3),
    "deletedAt" TIMESTAMP(3),
    "bootstrapUserId" TEXT,

    CONSTRAINT "Organization_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "OrganizationAlias" (
    "id" TEXT NOT NULL,
    "organizationId" TEXT NOT NULL,
    "aliasSlug" TEXT NOT NULL,
    "aliasType" "OrgAliasType" NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "createdByUserId" TEXT,
    "removedAt" TIMESTAMP(3),
    "removedByUserId" TEXT,
    "isRemovable" BOOLEAN NOT NULL DEFAULT true,

    CONSTRAINT "OrganizationAlias_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "OrganizationProfile" (
    "organizationId" TEXT NOT NULL,
    "username" TEXT NOT NULL,
    "displayName" TEXT,
    "avatarUrl" TEXT,
    "bio" TEXT,
    "socialLinks" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "OrganizationProfile_pkey" PRIMARY KEY ("organizationId")
);

-- CreateTable
CREATE TABLE "OrgMember" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "orgId" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "isOwner" BOOLEAN NOT NULL DEFAULT false,
    "isAdmin" BOOLEAN NOT NULL DEFAULT false,
    "isBillingManager" BOOLEAN NOT NULL DEFAULT false,
    "status" "OrgMemberStatus" NOT NULL DEFAULT 'ACTIVE',
    "joinedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "invitedByUserId" TEXT,

    CONSTRAINT "OrgMember_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "OrgInvitation" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "orgId" TEXT NOT NULL,
    "email" TEXT NOT NULL,
    "targetUserId" TEXT,
    "isAdmin" BOOLEAN NOT NULL DEFAULT false,
    "isBillingManager" BOOLEAN NOT NULL DEFAULT false,
    "token" TEXT NOT NULL,
    "tokenHash" TEXT,
    "expiresAt" TIMESTAMP(3) NOT NULL,
    "acceptedAt" TIMESTAMP(3),
    "revokedAt" TIMESTAMP(3),
    "invitedByUserId" TEXT NOT NULL,
    "workspaceIds" TEXT[],

    CONSTRAINT "OrgInvitation_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "OrgWorkspace" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "name" TEXT NOT NULL,
    "slug" TEXT,
    "description" TEXT,
    "isDefault" BOOLEAN NOT NULL DEFAULT false,
    "joinPolicy" "WorkspaceJoinPolicy" NOT NULL DEFAULT 'OPEN',
    "orgId" TEXT NOT NULL,
    "archivedAt" TIMESTAMP(3),
    "createdByUserId" TEXT,

    CONSTRAINT "OrgWorkspace_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "OrgWorkspaceMember" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "workspaceId" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "isAdmin" BOOLEAN NOT NULL DEFAULT false,
    "isBillingManager" BOOLEAN NOT NULL DEFAULT false,
    "status" "OrgMemberStatus" NOT NULL DEFAULT 'ACTIVE',
    "joinedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "invitedByUserId" TEXT,

    CONSTRAINT "OrgWorkspaceMember_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "WorkspaceInvite" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "workspaceId" TEXT NOT NULL,
    "email" TEXT NOT NULL,
    "targetUserId" TEXT,
    "isAdmin" BOOLEAN NOT NULL DEFAULT false,
    "isBillingManager" BOOLEAN NOT NULL DEFAULT false,
    "token" TEXT NOT NULL,
    "tokenHash" TEXT,
    "expiresAt" TIMESTAMP(3) NOT NULL,
    "acceptedAt" TIMESTAMP(3),
    "revokedAt" TIMESTAMP(3),
    "invitedByUserId" TEXT NOT NULL,

    CONSTRAINT "WorkspaceInvite_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "OrganizationSubscription" (
    "organizationId" TEXT NOT NULL,
    "planCode" TEXT,
    "planTier" TEXT,
    "stripeCustomerId" TEXT,
    "stripeSubscriptionId" TEXT,
    "status" TEXT NOT NULL DEFAULT 'active',
    "renewalAt" TIMESTAMP(3),
    "cancelAt" TIMESTAMP(3),
    "entitlements" JSONB NOT NULL DEFAULT '{}',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "OrganizationSubscription_pkey" PRIMARY KEY ("organizationId")
);

-- CreateTable
CREATE TABLE "OrganizationSeatAssignment" (
    "id" TEXT NOT NULL,
    "organizationId" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "seatType" "SeatType" NOT NULL DEFAULT 'FREE',
    "status" "SeatStatus" NOT NULL DEFAULT 'ACTIVE',
    "assignedByUserId" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "OrganizationSeatAssignment_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "OrgBalance" (
    "orgId" TEXT NOT NULL,
    "balance" INTEGER NOT NULL DEFAULT 0,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "OrgBalance_pkey" PRIMARY KEY ("orgId")
);

-- CreateTable
CREATE TABLE "OrgCreditTransaction" (
    "transactionKey" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "orgId" TEXT NOT NULL,
    "initiatedByUserId" TEXT,
    "workspaceId" TEXT,
    "amount" INTEGER NOT NULL,
    "type" "CreditTransactionType" NOT NULL,
    "runningBalance" INTEGER,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "metadata" JSONB,

    CONSTRAINT "OrgCreditTransaction_pkey" PRIMARY KEY ("transactionKey","orgId")
);

-- CreateTable
CREATE TABLE "TransferRequest" (
    "id" TEXT NOT NULL,
    "resourceType" TEXT NOT NULL,
    "resourceId" TEXT NOT NULL,
    "sourceOrganizationId" TEXT NOT NULL,
    "targetOrganizationId" TEXT NOT NULL,
    "initiatedByUserId" TEXT NOT NULL,
    "status" "TransferStatus" NOT NULL DEFAULT 'PENDING',
    "sourceApprovedByUserId" TEXT,
    "targetApprovedByUserId" TEXT,
    "completedAt" TIMESTAMP(3),
    "reason" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "TransferRequest_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "AuditLog" (
    "id" TEXT NOT NULL,
    "organizationId" TEXT,
    "workspaceId" TEXT,
    "actorUserId" TEXT NOT NULL,
    "entityType" TEXT NOT NULL,
    "entityId" TEXT,
    "action" TEXT NOT NULL,
    "beforeJson" JSONB,
    "afterJson" JSONB,
    "correlationId" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "AuditLog_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "IntegrationCredential" (
    "id" TEXT NOT NULL,
    "organizationId" TEXT NOT NULL,
    "ownerType" "CredentialOwnerType" NOT NULL,
    "ownerId" TEXT NOT NULL,
    "provider" TEXT NOT NULL,
    "credentialType" TEXT NOT NULL,
    "displayName" TEXT NOT NULL,
    "encryptedPayload" TEXT NOT NULL,
    "createdByUserId" TEXT NOT NULL,
    "lastUsedAt" TIMESTAMP(3),
    "status" TEXT NOT NULL DEFAULT 'active',
    "metadata" JSONB,
    "expiresAt" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "IntegrationCredential_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "Organization_slug_key" ON "Organization"("slug");

-- CreateIndex
CREATE INDEX "Organization_slug_idx" ON "Organization"("slug");

-- CreateIndex
CREATE UNIQUE INDEX "OrganizationAlias_aliasSlug_key" ON "OrganizationAlias"("aliasSlug");

-- CreateIndex
CREATE INDEX "OrganizationAlias_aliasSlug_idx" ON "OrganizationAlias"("aliasSlug");

-- CreateIndex
CREATE INDEX "OrganizationAlias_organizationId_idx" ON "OrganizationAlias"("organizationId");

-- CreateIndex
CREATE UNIQUE INDEX "OrganizationProfile_username_key" ON "OrganizationProfile"("username");

-- CreateIndex
CREATE INDEX "OrgMember_userId_idx" ON "OrgMember"("userId");

-- CreateIndex
CREATE INDEX "OrgMember_orgId_status_idx" ON "OrgMember"("orgId", "status");

-- CreateIndex
CREATE UNIQUE INDEX "OrgMember_orgId_userId_key" ON "OrgMember"("orgId", "userId");

-- CreateIndex
CREATE UNIQUE INDEX "OrgInvitation_token_key" ON "OrgInvitation"("token");

-- CreateIndex
CREATE INDEX "OrgInvitation_email_idx" ON "OrgInvitation"("email");

-- CreateIndex
CREATE INDEX "OrgInvitation_token_idx" ON "OrgInvitation"("token");

-- CreateIndex
CREATE INDEX "OrgInvitation_orgId_idx" ON "OrgInvitation"("orgId");

-- CreateIndex
CREATE INDEX "OrgWorkspace_orgId_isDefault_idx" ON "OrgWorkspace"("orgId", "isDefault");

-- CreateIndex
CREATE INDEX "OrgWorkspace_orgId_joinPolicy_idx" ON "OrgWorkspace"("orgId", "joinPolicy");

-- CreateIndex
CREATE UNIQUE INDEX "OrgWorkspace_orgId_name_key" ON "OrgWorkspace"("orgId", "name");

-- CreateIndex
CREATE INDEX "OrgWorkspaceMember_userId_idx" ON "OrgWorkspaceMember"("userId");

-- CreateIndex
CREATE INDEX "OrgWorkspaceMember_workspaceId_status_idx" ON "OrgWorkspaceMember"("workspaceId", "status");

-- CreateIndex
CREATE UNIQUE INDEX "OrgWorkspaceMember_workspaceId_userId_key" ON "OrgWorkspaceMember"("workspaceId", "userId");

-- CreateIndex
CREATE UNIQUE INDEX "WorkspaceInvite_token_key" ON "WorkspaceInvite"("token");

-- CreateIndex
CREATE INDEX "WorkspaceInvite_email_idx" ON "WorkspaceInvite"("email");

-- CreateIndex
CREATE INDEX "WorkspaceInvite_token_idx" ON "WorkspaceInvite"("token");

-- CreateIndex
CREATE INDEX "WorkspaceInvite_workspaceId_idx" ON "WorkspaceInvite"("workspaceId");

-- CreateIndex
CREATE INDEX "OrganizationSeatAssignment_organizationId_status_idx" ON "OrganizationSeatAssignment"("organizationId", "status");

-- CreateIndex
CREATE UNIQUE INDEX "OrganizationSeatAssignment_organizationId_userId_key" ON "OrganizationSeatAssignment"("organizationId", "userId");

-- CreateIndex
CREATE INDEX "OrgCreditTransaction_orgId_createdAt_idx" ON "OrgCreditTransaction"("orgId", "createdAt");

-- CreateIndex
CREATE INDEX "OrgCreditTransaction_initiatedByUserId_idx" ON "OrgCreditTransaction"("initiatedByUserId");

-- CreateIndex
CREATE INDEX "TransferRequest_sourceOrganizationId_idx" ON "TransferRequest"("sourceOrganizationId");

-- CreateIndex
CREATE INDEX "TransferRequest_targetOrganizationId_idx" ON "TransferRequest"("targetOrganizationId");

-- CreateIndex
CREATE INDEX "TransferRequest_status_idx" ON "TransferRequest"("status");

-- CreateIndex
CREATE INDEX "AuditLog_organizationId_createdAt_idx" ON "AuditLog"("organizationId", "createdAt");

-- CreateIndex
CREATE INDEX "AuditLog_actorUserId_createdAt_idx" ON "AuditLog"("actorUserId", "createdAt");

-- CreateIndex
CREATE INDEX "AuditLog_entityType_entityId_idx" ON "AuditLog"("entityType", "entityId");

-- CreateIndex
CREATE INDEX "IntegrationCredential_organizationId_ownerType_provider_idx" ON "IntegrationCredential"("organizationId", "ownerType", "provider");

-- CreateIndex
CREATE INDEX "IntegrationCredential_ownerId_ownerType_idx" ON "IntegrationCredential"("ownerId", "ownerType");

-- CreateIndex
CREATE INDEX "IntegrationCredential_createdByUserId_idx" ON "IntegrationCredential"("createdByUserId");

-- CreateIndex
CREATE INDEX "ChatSession_orgWorkspaceId_updatedAt_idx" ON "ChatSession"("orgWorkspaceId", "updatedAt");

-- CreateIndex
CREATE INDEX "AgentGraph_orgWorkspaceId_isActive_id_version_idx" ON "AgentGraph"("orgWorkspaceId", "isActive", "id", "version");

-- CreateIndex
CREATE INDEX "AgentPreset_orgWorkspaceId_idx" ON "AgentPreset"("orgWorkspaceId");

-- CreateIndex
CREATE INDEX "LibraryAgent_orgWorkspaceId_idx" ON "LibraryAgent"("orgWorkspaceId");

-- CreateIndex
CREATE INDEX "AgentGraphExecution_orgWorkspaceId_isDeleted_createdAt_idx" ON "AgentGraphExecution"("orgWorkspaceId", "isDeleted", "createdAt");

-- CreateIndex
CREATE INDEX "StoreListing_owningOrgId_idx" ON "StoreListing"("owningOrgId");

-- CreateIndex
CREATE INDEX "APIKey_orgWorkspaceId_idx" ON "APIKey"("orgWorkspaceId");

-- AddForeignKey
ALTER TABLE "ChatSession" ADD CONSTRAINT "ChatSession_orgWorkspaceId_fkey" FOREIGN KEY ("orgWorkspaceId") REFERENCES "OrgWorkspace"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AgentGraph" ADD CONSTRAINT "AgentGraph_orgWorkspaceId_fkey" FOREIGN KEY ("orgWorkspaceId") REFERENCES "OrgWorkspace"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AgentPreset" ADD CONSTRAINT "AgentPreset_orgWorkspaceId_fkey" FOREIGN KEY ("orgWorkspaceId") REFERENCES "OrgWorkspace"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LibraryAgent" ADD CONSTRAINT "LibraryAgent_orgWorkspaceId_fkey" FOREIGN KEY ("orgWorkspaceId") REFERENCES "OrgWorkspace"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LibraryFolder" ADD CONSTRAINT "LibraryFolder_orgWorkspaceId_fkey" FOREIGN KEY ("orgWorkspaceId") REFERENCES "OrgWorkspace"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AgentGraphExecution" ADD CONSTRAINT "AgentGraphExecution_orgWorkspaceId_fkey" FOREIGN KEY ("orgWorkspaceId") REFERENCES "OrgWorkspace"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "IntegrationWebhook" ADD CONSTRAINT "IntegrationWebhook_orgWorkspaceId_fkey" FOREIGN KEY ("orgWorkspaceId") REFERENCES "OrgWorkspace"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "StoreListing" ADD CONSTRAINT "StoreListing_owningOrgId_fkey" FOREIGN KEY ("owningOrgId") REFERENCES "Organization"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "APIKey" ADD CONSTRAINT "APIKey_orgWorkspaceId_fkey" FOREIGN KEY ("orgWorkspaceId") REFERENCES "OrgWorkspace"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "OrganizationAlias" ADD CONSTRAINT "OrganizationAlias_organizationId_fkey" FOREIGN KEY ("organizationId") REFERENCES "Organization"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "OrganizationProfile" ADD CONSTRAINT "OrganizationProfile_organizationId_fkey" FOREIGN KEY ("organizationId") REFERENCES "Organization"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "OrgMember" ADD CONSTRAINT "OrgMember_orgId_fkey" FOREIGN KEY ("orgId") REFERENCES "Organization"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "OrgMember" ADD CONSTRAINT "OrgMember_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "OrgInvitation" ADD CONSTRAINT "OrgInvitation_orgId_fkey" FOREIGN KEY ("orgId") REFERENCES "Organization"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "OrgWorkspace" ADD CONSTRAINT "OrgWorkspace_orgId_fkey" FOREIGN KEY ("orgId") REFERENCES "Organization"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "OrgWorkspaceMember" ADD CONSTRAINT "OrgWorkspaceMember_workspaceId_fkey" FOREIGN KEY ("workspaceId") REFERENCES "OrgWorkspace"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "OrgWorkspaceMember" ADD CONSTRAINT "OrgWorkspaceMember_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "OrganizationSubscription" ADD CONSTRAINT "OrganizationSubscription_organizationId_fkey" FOREIGN KEY ("organizationId") REFERENCES "Organization"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "OrganizationSeatAssignment" ADD CONSTRAINT "OrganizationSeatAssignment_organizationId_userId_fkey" FOREIGN KEY ("organizationId", "userId") REFERENCES "OrgMember"("orgId", "userId") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "OrganizationSeatAssignment" ADD CONSTRAINT "OrganizationSeatAssignment_organizationId_fkey" FOREIGN KEY ("organizationId") REFERENCES "Organization"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "OrgBalance" ADD CONSTRAINT "OrgBalance_orgId_fkey" FOREIGN KEY ("orgId") REFERENCES "Organization"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "OrgCreditTransaction" ADD CONSTRAINT "OrgCreditTransaction_orgId_fkey" FOREIGN KEY ("orgId") REFERENCES "Organization"("id") ON DELETE NO ACTION ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "TransferRequest" ADD CONSTRAINT "TransferRequest_sourceOrganizationId_fkey" FOREIGN KEY ("sourceOrganizationId") REFERENCES "Organization"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "TransferRequest" ADD CONSTRAINT "TransferRequest_targetOrganizationId_fkey" FOREIGN KEY ("targetOrganizationId") REFERENCES "Organization"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AuditLog" ADD CONSTRAINT "AuditLog_organizationId_fkey" FOREIGN KEY ("organizationId") REFERENCES "Organization"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "IntegrationCredential" ADD CONSTRAINT "IntegrationCredential_organizationId_fkey" FOREIGN KEY ("organizationId") REFERENCES "Organization"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "IntegrationCredential" ADD CONSTRAINT "IntegrationCredential_workspace_fkey" FOREIGN KEY ("ownerId") REFERENCES "OrgWorkspace"("id") ON DELETE CASCADE ON UPDATE CASCADE;

