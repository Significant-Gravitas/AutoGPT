BEGIN;

-- CreateEnum
CREATE TYPE "UserGroupRole" AS ENUM ('MEMBER', 'OWNER');

-- CreateEnum
CREATE TYPE "AgentExecutionStatus" AS ENUM ('INCOMPLETE', 'QUEUED', 'RUNNING', 'COMPLETED', 'FAILED');

-- CreateEnum
CREATE TYPE "ExecutionTriggerType" AS ENUM ('MANUAL', 'SCHEDULE', 'WEBHOOK');

-- CreateEnum
CREATE TYPE "HttpMethod" AS ENUM ('GET', 'POST', 'PUT', 'DELETE', 'PATCH');

-- CreateEnum
CREATE TYPE "UserBlockCreditType" AS ENUM ('TOP_UP', 'USAGE');

-- CreateEnum
CREATE TYPE "SubmissionStatus" AS ENUM ('DAFT', 'PENDING', 'APPROVED', 'REJECTED');

----------------------------------------------------------------------------------------------------
-- Table: AgentBlock
----------------------------------------------------------------------------------------------------

-- CreateTable
CREATE TABLE "AgentBlock" (
    "id" UUID NOT NULL,
    "name" TEXT NOT NULL,
    "inputSchema" JSONB NOT NULL DEFAULT '{}',
    "outputSchema" JSONB NOT NULL DEFAULT '{}',

    CONSTRAINT "AgentBlock_pkey" PRIMARY KEY ("id")
);

----------------------------------------------------------------------------------------------------
-- Table: User
----------------------------------------------------------------------------------------------------

-- CreateTable
CREATE TABLE "User" (
    "id" UUID NOT NULL,
    "email" TEXT NOT NULL,
    "name" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "metadata" JSONB DEFAULT '{}',

    CONSTRAINT "User_pkey" PRIMARY KEY ("id")
);

----------------------------------------------------------------------------------------------------
-- Table: AnalyticsDetails
----------------------------------------------------------------------------------------------------

-- CreateTable
CREATE TABLE "AnalyticsDetails" (
    "id" UUID NOT NULL DEFAULT gen_random_uuid(),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" UUID NOT NULL,
    "type" TEXT NOT NULL,
    "data" JSONB NOT NULL DEFAULT '{}',
    "dataIndex" TEXT,

    CONSTRAINT "AnalyticsDetails_pkey" PRIMARY KEY ("id")
);

----------------------------------------------------------------------------------------------------
-- Table: AnalyticsMetrics
----------------------------------------------------------------------------------------------------

-- CreateTable
CREATE TABLE "AnalyticsMetrics" (
    "id" UUID NOT NULL DEFAULT gen_random_uuid(),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "analyticMetric" TEXT NOT NULL,
    "value" DOUBLE PRECISION NOT NULL,
    "dataString" TEXT,
    "userId" UUID NOT NULL,

    CONSTRAINT "AnalyticsMetrics_pkey" PRIMARY KEY ("id")
);

----------------------------------------------------------------------------------------------------
-- Table: UserGroup
----------------------------------------------------------------------------------------------------

-- CreateTable
CREATE TABLE "UserGroup" (
    "id" UUID NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "name" TEXT NOT NULL,
    "description" TEXT NOT NULL,
    "groupIconUrl" TEXT,

    CONSTRAINT "UserGroup_pkey" PRIMARY KEY ("id")
);

----------------------------------------------------------------------------------------------------
-- Table: Agent
----------------------------------------------------------------------------------------------------

-- CreateTable
CREATE TABLE "Agent" (
    "id" UUID NOT NULL,
    "version" INTEGER NOT NULL DEFAULT 1,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "name" TEXT,
    "description" TEXT,
    "createdByUserId" UUID,
    "groupId" UUID,
    "agentParentId" UUID,
    "agentParentVersion" INTEGER,

    CONSTRAINT "Agent_pkey" PRIMARY KEY ("id","version")
);

----------------------------------------------------------------------------------------------------
-- Table: AgentNode
----------------------------------------------------------------------------------------------------

-- CreateTable
CREATE TABLE "AgentNode" (
    "id" UUID NOT NULL,
    "agentBlockId" UUID NOT NULL,
    "agentId" UUID NOT NULL,
    "agentVersion" INTEGER NOT NULL DEFAULT 1,
    "constantInput" JSONB NOT NULL DEFAULT '{}',
    "metadata" JSONB NOT NULL DEFAULT '{}',

    CONSTRAINT "AgentNode_pkey" PRIMARY KEY ("id")
);

----------------------------------------------------------------------------------------------------
-- Table: AgentNodeLink
----------------------------------------------------------------------------------------------------

-- CreateTable
CREATE TABLE "AgentNodeLink" (
    "id" UUID NOT NULL,
    "agentNodeSourceId" UUID NOT NULL,
    "sourceName" TEXT NOT NULL,
    "agentNodeSinkId" UUID NOT NULL,
    "sinkName" TEXT NOT NULL,
    "isStatic" BOOLEAN NOT NULL DEFAULT false,

    CONSTRAINT "AgentNodeLink_pkey" PRIMARY KEY ("id")
);

----------------------------------------------------------------------------------------------------
-- Table: AgentPreset
----------------------------------------------------------------------------------------------------

-- CreateTable
CREATE TABLE "AgentPreset" (
    "id" UUID NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "name" TEXT NOT NULL,
    "description" TEXT NOT NULL,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "userId" UUID NOT NULL,
    "agentId" UUID NOT NULL,
    "agentVersion" INTEGER NOT NULL,

    CONSTRAINT "AgentPreset_pkey" PRIMARY KEY ("id")
);

----------------------------------------------------------------------------------------------------
-- Table: AgentExecution
----------------------------------------------------------------------------------------------------

-- CreateTable
CREATE TABLE "AgentExecution" (
    "id" UUID NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "startedAt" TIMESTAMP(3),
    "executionTriggerType" "ExecutionTriggerType" NOT NULL DEFAULT 'MANUAL',
    "executionStatus" "AgentExecutionStatus" NOT NULL DEFAULT 'COMPLETED',
    "agentId" UUID NOT NULL,
    "agentVersion" INTEGER NOT NULL DEFAULT 1,
    "agentPresetId" UUID,
    "executedByUserId" UUID NOT NULL,
    "stats" JSONB NOT NULL DEFAULT '{}',

    CONSTRAINT "AgentExecution_pkey" PRIMARY KEY ("id")
);

----------------------------------------------------------------------------------------------------
-- Table: AgentExecutionSchedule
----------------------------------------------------------------------------------------------------

-- CreateTable
CREATE TABLE "AgentExecutionSchedule" (
    "id" UUID NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "agentPresetId" UUID NOT NULL,
    "schedule" TEXT NOT NULL,
    "isEnabled" BOOLEAN NOT NULL DEFAULT true,
    "triggerIdentifier" TEXT NOT NULL,
    "lastUpdated" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" UUID NOT NULL,
    "agentId" UUID,
    "agentVersion" INTEGER,

    CONSTRAINT "AgentExecutionSchedule_pkey" PRIMARY KEY ("id")
);

----------------------------------------------------------------------------------------------------
-- Table: AgentNodeExecution
----------------------------------------------------------------------------------------------------

-- CreateTable
CREATE TABLE "AgentNodeExecution" (
    "id" UUID NOT NULL,
    "agentExecutionId" UUID NOT NULL,
    "agentNodeId" UUID NOT NULL,
    "executionStatus" "AgentExecutionStatus" NOT NULL DEFAULT 'COMPLETED',
    "executionData" TEXT,
    "addedTime" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "queuedTime" TIMESTAMP(3),
    "startedTime" TIMESTAMP(3),
    "endedTime" TIMESTAMP(3),
    "stats" JSONB NOT NULL DEFAULT '{}',

    CONSTRAINT "AgentNodeExecution_pkey" PRIMARY KEY ("id")
);

----------------------------------------------------------------------------------------------------
-- Table: AgentNodeExecutionInputOutput
----------------------------------------------------------------------------------------------------

-- CreateTable
CREATE TABLE "AgentNodeExecutionInputOutput" (
    "id" UUID NOT NULL,
    "name" TEXT NOT NULL,
    "data" TEXT NOT NULL,
    "time" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "referencedByInputExecId" UUID,
    "referencedByOutputExecId" UUID,
    "agentPresetId" UUID,

    CONSTRAINT "AgentNodeExecutionInputOutput_pkey" PRIMARY KEY ("id")
);

----------------------------------------------------------------------------------------------------
-- Table: Profile
----------------------------------------------------------------------------------------------------

-- CreateTable
CREATE TABLE "Profile" (
    "id" UUID NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" UUID,
    "groupId" UUID,
    "username" TEXT NOT NULL,
    "description" TEXT NOT NULL,
    "links" TEXT[],
    "avatarUrl" TEXT,

    CONSTRAINT "Profile_pkey" PRIMARY KEY ("id")
);

----------------------------------------------------------------------------------------------------
-- Table: StoreListing
----------------------------------------------------------------------------------------------------

-- CreateTable
CREATE TABLE "StoreListing" (
    "id" UUID NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "isDeleted" BOOLEAN NOT NULL DEFAULT false,
    "isApproved" BOOLEAN NOT NULL DEFAULT false,
    "agentId" UUID NOT NULL,
    "agentVersion" INTEGER NOT NULL,
    "owningUserId" UUID NOT NULL,
    "isGroupListing" BOOLEAN NOT NULL DEFAULT false,
    "owningGroupId" UUID,

    CONSTRAINT "StoreListing_pkey" PRIMARY KEY ("id")
);

----------------------------------------------------------------------------------------------------
-- Table: StoreListingVersion
----------------------------------------------------------------------------------------------------

-- CreateTable
CREATE TABLE "StoreListingVersion" (
    "id" UUID NOT NULL,
    "version" INTEGER NOT NULL DEFAULT 1,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "agentId" UUID NOT NULL,
    "agentVersion" INTEGER NOT NULL,
    "slug" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "videoUrl" TEXT,
    "imageUrls" TEXT[],
    "description" TEXT NOT NULL,
    "categories" TEXT[],
    "isFeatured" BOOLEAN NOT NULL DEFAULT false,
    "isDeleted" BOOLEAN NOT NULL DEFAULT false,
    "isAvailable" BOOLEAN NOT NULL DEFAULT true,
    "isApproved" BOOLEAN NOT NULL DEFAULT false,
    "storeListingId" UUID,

    CONSTRAINT "StoreListingVersion_pkey" PRIMARY KEY ("id")
);

----------------------------------------------------------------------------------------------------
-- Table: StoreListingReview
----------------------------------------------------------------------------------------------------

-- CreateTable
CREATE TABLE "StoreListingReview" (
    "id" UUID NOT NULL DEFAULT gen_random_uuid(),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "storeListingVersionId" UUID NOT NULL,
    "reviewByUserId" UUID NOT NULL,
    "score" INTEGER NOT NULL,
    "comments" TEXT,

    CONSTRAINT "StoreListingReview_pkey" PRIMARY KEY ("id")
);

----------------------------------------------------------------------------------------------------
-- Table: StoreListingSubmission
----------------------------------------------------------------------------------------------------

-- CreateTable
CREATE TABLE "StoreListingSubmission" (
    "id" UUID NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "storeListingId" UUID NOT NULL,
    "storeListingVersionId" UUID NOT NULL,
    "reviewerId" UUID NOT NULL,
    "Status" "SubmissionStatus" NOT NULL DEFAULT 'PENDING',
    "reviewComments" TEXT,

    CONSTRAINT "StoreListingSubmission_pkey" PRIMARY KEY ("id")
);

----------------------------------------------------------------------------------------------------
-- Table: UserAgent
----------------------------------------------------------------------------------------------------

-- CreateTable
CREATE TABLE "UserAgent" (
    "id" UUID NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" UUID NOT NULL,
    "agentId" UUID NOT NULL,
    "agentVersion" INTEGER NOT NULL,
    "agentPresetId" UUID,
    "isFavorite" BOOLEAN NOT NULL DEFAULT false,
    "isCreatedByUser" BOOLEAN NOT NULL DEFAULT false,
    "isArchived" BOOLEAN NOT NULL DEFAULT false,
    "isDeleted" BOOLEAN NOT NULL DEFAULT false,

    CONSTRAINT "UserAgent_pkey" PRIMARY KEY ("id")
);

----------------------------------------------------------------------------------------------------
-- Table: UserBlockCredit
----------------------------------------------------------------------------------------------------

-- CreateTable
CREATE TABLE "UserBlockCredit" (
    "transactionKey" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" UUID NOT NULL,
    "blockId" UUID,
    "executedAgentId" UUID,
    "executedAgentVersion" INTEGER,
    "agentNodeExecutionId" UUID,
    "amount" INTEGER NOT NULL,
    "type" "UserBlockCreditType" NOT NULL,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "metadata" JSONB NOT NULL DEFAULT '{}',

    CONSTRAINT "UserBlockCredit_pkey" PRIMARY KEY ("transactionKey","userId")
);

----------------------------------------------------------------------------------------------------
-- Table: UserGroupMembership
----------------------------------------------------------------------------------------------------

-- CreateTable
CREATE TABLE "UserGroupMembership" (
    "id" UUID NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" UUID NOT NULL,
    "userGroupId" UUID NOT NULL,
    "Role" "UserGroupRole" NOT NULL DEFAULT 'MEMBER',

    CONSTRAINT "UserGroupMembership_pkey" PRIMARY KEY ("id")
);

----------------------------------------------------------------------------------------------------
-- Table: WebhookTrigger
----------------------------------------------------------------------------------------------------

-- CreateTable
CREATE TABLE "WebhookTrigger" (
    "id" UUID NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "agentPresetId" UUID NOT NULL,
    "method" "HttpMethod" NOT NULL,
    "urlSlug" TEXT NOT NULL,
    "triggerIdentifier" TEXT NOT NULL,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "lastReceivedDataAt" TIMESTAMP(3),
    "isDeleted" BOOLEAN NOT NULL DEFAULT false,
    "agentId" UUID,
    "agentVersion" INTEGER,

    CONSTRAINT "WebhookTrigger_pkey" PRIMARY KEY ("id")
);

----------------------------------------------------------------------------------------------------

-- Deferred Foreign Key Constraints (Cyclic Dependencies)
----------------------------------------------------------------------------------------------------

-- AddForeignKey
ALTER TABLE "Agent" ADD CONSTRAINT "Agent_agentParentId_agentParentVersion_fkey" FOREIGN KEY ("agentParentId", "agentParentVersion") REFERENCES "Agent"("id", "version") ON DELETE SET NULL ON UPDATE CASCADE;

COMMIT;