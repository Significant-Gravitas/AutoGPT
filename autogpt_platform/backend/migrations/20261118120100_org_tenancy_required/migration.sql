-- Every row of these tables now belongs to an organization: step 1 of 2.
-- The app's startup bootstrap (backend/data/org_migration.py) gives untagged
-- rows their owner's personal organization and its default team; this tags
-- any it has not reached yet the same way, refuses to go on if a row's owner
-- has no personal organization, and adds each NOT NULL as a NOT VALID check.
-- 20261118120200 validates the checks and sets the columns NOT NULL in its own
-- transaction, so the scan of existing rows holds no lock that blocks writes.
-- teamId stays nullable: NULL is an org-home row, and deleting a team sets it.

-- CreateIndex
-- Declared in schema.prisma since the org tables arrived, never created by a
-- migration; IF NOT EXISTS skips any built out of band.
CREATE INDEX IF NOT EXISTS "AgentGraph_organizationId_idx" ON "AgentGraph"("organizationId");
CREATE INDEX IF NOT EXISTS "ChatSession_organizationId_idx" ON "ChatSession"("organizationId");
CREATE INDEX IF NOT EXISTS "AgentPreset_organizationId_idx" ON "AgentPreset"("organizationId");
CREATE INDEX IF NOT EXISTS "LibraryAgent_organizationId_idx" ON "LibraryAgent"("organizationId");
CREATE INDEX IF NOT EXISTS "LibraryFolder_organizationId_idx" ON "LibraryFolder"("organizationId");
CREATE INDEX IF NOT EXISTS "IntegrationWebhook_organizationId_idx" ON "IntegrationWebhook"("organizationId");
CREATE INDEX IF NOT EXISTS "APIKey_organizationId_idx" ON "APIKey"("organizationId");
CREATE INDEX IF NOT EXISTS "UserNotificationBatch_organizationId_idx" ON "UserNotificationBatch"("organizationId");
CREATE INDEX IF NOT EXISTS "BuilderSearchHistory_organizationId_idx" ON "BuilderSearchHistory"("organizationId");
CREATE INDEX IF NOT EXISTS "PendingHumanReview_organizationId_idx" ON "PendingHumanReview"("organizationId");
CREATE INDEX IF NOT EXISTS "StoreListingVersion_organizationId_idx" ON "StoreListingVersion"("organizationId");
CREATE INDEX IF NOT EXISTS "OAuthApplication_organizationId_idx" ON "OAuthApplication"("organizationId");
CREATE INDEX IF NOT EXISTS "OrgCreditTransaction_teamId_idx" ON "OrgCreditTransaction"("teamId");

-- Tag rows the startup bootstrap has not reached with the owner's personal org.
UPDATE "AgentGraph" t
SET "organizationId" = o."id", "teamId" = w."id"
FROM "OrgMember" om
JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" AND o."deletedAt" IS NULL
JOIN "Team" w ON w."orgId" = o."id" AND w."isDefault"
WHERE t."userId" = om."userId" AND om."isOwner" AND t."organizationId" IS NULL;
UPDATE "AgentGraphExecution" t
SET "organizationId" = o."id", "teamId" = w."id"
FROM "OrgMember" om
JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" AND o."deletedAt" IS NULL
JOIN "Team" w ON w."orgId" = o."id" AND w."isDefault"
WHERE t."userId" = om."userId" AND om."isOwner" AND t."organizationId" IS NULL;
UPDATE "ChatSession" t
SET "organizationId" = o."id", "teamId" = w."id"
FROM "OrgMember" om
JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" AND o."deletedAt" IS NULL
JOIN "Team" w ON w."orgId" = o."id" AND w."isDefault"
WHERE t."userId" = om."userId" AND om."isOwner" AND t."organizationId" IS NULL;
UPDATE "AgentPreset" t
SET "organizationId" = o."id", "teamId" = w."id"
FROM "OrgMember" om
JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" AND o."deletedAt" IS NULL
JOIN "Team" w ON w."orgId" = o."id" AND w."isDefault"
WHERE t."userId" = om."userId" AND om."isOwner" AND t."organizationId" IS NULL;
UPDATE "LibraryAgent" t
SET "organizationId" = o."id", "teamId" = w."id"
FROM "OrgMember" om
JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" AND o."deletedAt" IS NULL
JOIN "Team" w ON w."orgId" = o."id" AND w."isDefault"
WHERE t."userId" = om."userId" AND om."isOwner" AND t."organizationId" IS NULL;
UPDATE "LibraryFolder" t
SET "organizationId" = o."id", "teamId" = w."id"
FROM "OrgMember" om
JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" AND o."deletedAt" IS NULL
JOIN "Team" w ON w."orgId" = o."id" AND w."isDefault"
WHERE t."userId" = om."userId" AND om."isOwner" AND t."organizationId" IS NULL;
UPDATE "IntegrationWebhook" t
SET "organizationId" = o."id", "teamId" = w."id"
FROM "OrgMember" om
JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" AND o."deletedAt" IS NULL
JOIN "Team" w ON w."orgId" = o."id" AND w."isDefault"
WHERE t."userId" = om."userId" AND om."isOwner" AND t."organizationId" IS NULL;
UPDATE "APIKey" t
SET "organizationId" = o."id", "teamId" = w."id"
FROM "OrgMember" om
JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" AND o."deletedAt" IS NULL
JOIN "Team" w ON w."orgId" = o."id" AND w."isDefault"
WHERE t."userId" = om."userId" AND om."isOwner" AND t."organizationId" IS NULL;
UPDATE "UserNotificationBatch" t
SET "organizationId" = o."id", "teamId" = w."id"
FROM "OrgMember" om
JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" AND o."deletedAt" IS NULL
JOIN "Team" w ON w."orgId" = o."id" AND w."isDefault"
WHERE t."userId" = om."userId" AND om."isOwner" AND t."organizationId" IS NULL;
UPDATE "BuilderSearchHistory" t
SET "organizationId" = o."id"
FROM "OrgMember" om
JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" AND o."deletedAt" IS NULL
WHERE t."userId" = om."userId" AND om."isOwner" AND t."organizationId" IS NULL;
UPDATE "PendingHumanReview" t
SET "organizationId" = o."id"
FROM "OrgMember" om
JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" AND o."deletedAt" IS NULL
WHERE t."userId" = om."userId" AND om."isOwner" AND t."organizationId" IS NULL;
UPDATE "StoreListing" sl
SET "owningOrgId" = o."id"
FROM "OrgMember" om
JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" AND o."deletedAt" IS NULL
WHERE sl."owningUserId" = om."userId" AND om."isOwner" AND sl."owningOrgId" IS NULL;
UPDATE "StoreListingVersion" v
SET "organizationId" = sl."owningOrgId"
FROM "StoreListing" sl
WHERE sl."id" = v."storeListingId" AND v."organizationId" IS NULL AND sl."owningOrgId" IS NOT NULL;

-- A row still untagged here belongs to a user with no personal org, which only
-- the app's startup creates: an install upgrading straight from a release
-- before organizations. Stop rather than guess one.
DO $$
DECLARE
    remaining text;
BEGIN
    SELECT string_agg(format('%s: %s', tbl, n), ', ') INTO remaining
    FROM (
        SELECT 'AgentGraph', count(*) FROM "AgentGraph" WHERE "organizationId" IS NULL
        UNION ALL SELECT 'AgentGraphExecution', count(*) FROM "AgentGraphExecution" WHERE "organizationId" IS NULL
        UNION ALL SELECT 'ChatSession', count(*) FROM "ChatSession" WHERE "organizationId" IS NULL
        UNION ALL SELECT 'AgentPreset', count(*) FROM "AgentPreset" WHERE "organizationId" IS NULL
        UNION ALL SELECT 'LibraryAgent', count(*) FROM "LibraryAgent" WHERE "organizationId" IS NULL
        UNION ALL SELECT 'LibraryFolder', count(*) FROM "LibraryFolder" WHERE "organizationId" IS NULL
        UNION ALL SELECT 'IntegrationWebhook', count(*) FROM "IntegrationWebhook" WHERE "organizationId" IS NULL
        UNION ALL SELECT 'APIKey', count(*) FROM "APIKey" WHERE "organizationId" IS NULL
        UNION ALL SELECT 'UserNotificationBatch', count(*) FROM "UserNotificationBatch" WHERE "organizationId" IS NULL
        UNION ALL SELECT 'BuilderSearchHistory', count(*) FROM "BuilderSearchHistory" WHERE "organizationId" IS NULL
        UNION ALL SELECT 'PendingHumanReview', count(*) FROM "PendingHumanReview" WHERE "organizationId" IS NULL
        UNION ALL SELECT 'StoreListingVersion', count(*) FROM "StoreListingVersion" WHERE "organizationId" IS NULL
        UNION ALL SELECT 'StoreListing', count(*) FROM "StoreListing" WHERE "owningOrgId" IS NULL
    ) AS untagged(tbl, n)
    WHERE n > 0;
    IF remaining IS NOT NULL THEN
        RAISE EXCEPTION 'Rows without an organization remain (%). Their owners have no personal organization yet. Mark this migration rolled back (prisma migrate resolve --rolled-back 20261118120100_org_tenancy_required), start the previous release once so its startup creates them, then deploy this one again.', remaining;
    END IF;
END $$;

-- AddCheckConstraint
-- NOT VALID: checked for new rows from here on, existing rows in 20261118120200.
ALTER TABLE "AgentGraph" ADD CONSTRAINT "AgentGraph_organizationId_not_null" CHECK ("organizationId" IS NOT NULL) NOT VALID;
ALTER TABLE "AgentGraphExecution" ADD CONSTRAINT "AgentGraphExecution_organizationId_not_null" CHECK ("organizationId" IS NOT NULL) NOT VALID;
ALTER TABLE "ChatSession" ADD CONSTRAINT "ChatSession_organizationId_not_null" CHECK ("organizationId" IS NOT NULL) NOT VALID;
ALTER TABLE "AgentPreset" ADD CONSTRAINT "AgentPreset_organizationId_not_null" CHECK ("organizationId" IS NOT NULL) NOT VALID;
ALTER TABLE "LibraryAgent" ADD CONSTRAINT "LibraryAgent_organizationId_not_null" CHECK ("organizationId" IS NOT NULL) NOT VALID;
ALTER TABLE "LibraryFolder" ADD CONSTRAINT "LibraryFolder_organizationId_not_null" CHECK ("organizationId" IS NOT NULL) NOT VALID;
ALTER TABLE "IntegrationWebhook" ADD CONSTRAINT "IntegrationWebhook_organizationId_not_null" CHECK ("organizationId" IS NOT NULL) NOT VALID;
ALTER TABLE "APIKey" ADD CONSTRAINT "APIKey_organizationId_not_null" CHECK ("organizationId" IS NOT NULL) NOT VALID;
ALTER TABLE "UserNotificationBatch" ADD CONSTRAINT "UserNotificationBatch_organizationId_not_null" CHECK ("organizationId" IS NOT NULL) NOT VALID;
ALTER TABLE "BuilderSearchHistory" ADD CONSTRAINT "BuilderSearchHistory_organizationId_not_null" CHECK ("organizationId" IS NOT NULL) NOT VALID;
ALTER TABLE "PendingHumanReview" ADD CONSTRAINT "PendingHumanReview_organizationId_not_null" CHECK ("organizationId" IS NOT NULL) NOT VALID;
ALTER TABLE "StoreListingVersion" ADD CONSTRAINT "StoreListingVersion_organizationId_not_null" CHECK ("organizationId" IS NOT NULL) NOT VALID;
ALTER TABLE "StoreListing" ADD CONSTRAINT "StoreListing_owningOrgId_not_null" CHECK ("owningOrgId" IS NOT NULL) NOT VALID;

-- CreateIndex
-- A listing's slug is unique within its organization. The per-user key stays:
-- the marketplace still finds a listing by its creator's username and slug.
CREATE UNIQUE INDEX "StoreListing_owningOrgId_slug_key" ON "StoreListing"("owningOrgId", "slug");
