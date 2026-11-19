-- Every row of these tables now belongs to an organization: step 2 of 2.
-- Validates the checks 20261118120100 added NOT VALID, in this transaction of
-- its own so the scans hold a SHARE UPDATE EXCLUSIVE lock and reads and writes
-- go on. With a validated check in place SET NOT NULL skips its own scan, so
-- the ACCESS EXCLUSIVE locks at the end are held only for catalog updates.

-- A VALIDATE scan of a large table can outlast the role's statement timeout,
-- and holds no lock that blocks reads or writes. The locks SET NOT NULL takes
-- do, so it fails rather than queue behind a long transaction.
SET LOCAL statement_timeout = '30min';
SET LOCAL lock_timeout = '10s';

-- ValidateCheckConstraint
ALTER TABLE "AgentGraph" VALIDATE CONSTRAINT "AgentGraph_organizationId_not_null";
ALTER TABLE "AgentGraphExecution" VALIDATE CONSTRAINT "AgentGraphExecution_organizationId_not_null";
ALTER TABLE "ChatSession" VALIDATE CONSTRAINT "ChatSession_organizationId_not_null";
ALTER TABLE "AgentPreset" VALIDATE CONSTRAINT "AgentPreset_organizationId_not_null";
ALTER TABLE "LibraryAgent" VALIDATE CONSTRAINT "LibraryAgent_organizationId_not_null";
ALTER TABLE "LibraryFolder" VALIDATE CONSTRAINT "LibraryFolder_organizationId_not_null";
ALTER TABLE "IntegrationWebhook" VALIDATE CONSTRAINT "IntegrationWebhook_organizationId_not_null";
ALTER TABLE "APIKey" VALIDATE CONSTRAINT "APIKey_organizationId_not_null";
ALTER TABLE "UserNotificationBatch" VALIDATE CONSTRAINT "UserNotificationBatch_organizationId_not_null";
ALTER TABLE "BuilderSearchHistory" VALIDATE CONSTRAINT "BuilderSearchHistory_organizationId_not_null";
ALTER TABLE "PendingHumanReview" VALIDATE CONSTRAINT "PendingHumanReview_organizationId_not_null";
ALTER TABLE "StoreListingVersion" VALIDATE CONSTRAINT "StoreListingVersion_organizationId_not_null";
ALTER TABLE "StoreListing" VALIDATE CONSTRAINT "StoreListing_owningOrgId_not_null";

-- AlterTable
ALTER TABLE "AgentGraph" ALTER COLUMN "organizationId" SET NOT NULL;
ALTER TABLE "AgentGraphExecution" ALTER COLUMN "organizationId" SET NOT NULL;
ALTER TABLE "ChatSession" ALTER COLUMN "organizationId" SET NOT NULL;
ALTER TABLE "AgentPreset" ALTER COLUMN "organizationId" SET NOT NULL;
ALTER TABLE "LibraryAgent" ALTER COLUMN "organizationId" SET NOT NULL;
ALTER TABLE "LibraryFolder" ALTER COLUMN "organizationId" SET NOT NULL;
ALTER TABLE "IntegrationWebhook" ALTER COLUMN "organizationId" SET NOT NULL;
ALTER TABLE "APIKey" ALTER COLUMN "organizationId" SET NOT NULL;
ALTER TABLE "UserNotificationBatch" ALTER COLUMN "organizationId" SET NOT NULL;
ALTER TABLE "BuilderSearchHistory" ALTER COLUMN "organizationId" SET NOT NULL;
ALTER TABLE "PendingHumanReview" ALTER COLUMN "organizationId" SET NOT NULL;
ALTER TABLE "StoreListingVersion" ALTER COLUMN "organizationId" SET NOT NULL;
ALTER TABLE "StoreListing" ALTER COLUMN "owningOrgId" SET NOT NULL;

-- DropCheckConstraint
-- Only now: in the SET NOT NULL statement itself the drop runs first, and the
-- column is then scanned again under its ACCESS EXCLUSIVE lock.
ALTER TABLE "AgentGraph" DROP CONSTRAINT "AgentGraph_organizationId_not_null";
ALTER TABLE "AgentGraphExecution" DROP CONSTRAINT "AgentGraphExecution_organizationId_not_null";
ALTER TABLE "ChatSession" DROP CONSTRAINT "ChatSession_organizationId_not_null";
ALTER TABLE "AgentPreset" DROP CONSTRAINT "AgentPreset_organizationId_not_null";
ALTER TABLE "LibraryAgent" DROP CONSTRAINT "LibraryAgent_organizationId_not_null";
ALTER TABLE "LibraryFolder" DROP CONSTRAINT "LibraryFolder_organizationId_not_null";
ALTER TABLE "IntegrationWebhook" DROP CONSTRAINT "IntegrationWebhook_organizationId_not_null";
ALTER TABLE "APIKey" DROP CONSTRAINT "APIKey_organizationId_not_null";
ALTER TABLE "UserNotificationBatch" DROP CONSTRAINT "UserNotificationBatch_organizationId_not_null";
ALTER TABLE "BuilderSearchHistory" DROP CONSTRAINT "BuilderSearchHistory_organizationId_not_null";
ALTER TABLE "PendingHumanReview" DROP CONSTRAINT "PendingHumanReview_organizationId_not_null";
ALTER TABLE "StoreListingVersion" DROP CONSTRAINT "StoreListingVersion_organizationId_not_null";
ALTER TABLE "StoreListing" DROP CONSTRAINT "StoreListing_owningOrgId_not_null";

-- AlterForeignKey
-- A required relation: an organization that owns listings can't be deleted
-- out from under them (SET NULL would now fail the NOT NULL anyway).
ALTER TABLE "StoreListing" DROP CONSTRAINT "StoreListing_owningOrgId_fkey";
ALTER TABLE "StoreListing" ADD CONSTRAINT "StoreListing_owningOrgId_fkey" FOREIGN KEY ("owningOrgId") REFERENCES "Organization"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
