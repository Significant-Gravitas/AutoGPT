WITH ranked_owners AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            PARTITION BY "orgId"
            ORDER BY "updatedAt" DESC, "createdAt" DESC, id DESC
        ) AS row_number
    FROM "OrgMember"
    WHERE "isOwner" = true
)
UPDATE "OrgMember"
SET "isOwner" = false,
    "updatedAt" = NOW()
WHERE id IN (
    SELECT id FROM ranked_owners WHERE row_number > 1
);

CREATE UNIQUE INDEX IF NOT EXISTS "OrgMember_one_owner_per_org"
ON "OrgMember" ("orgId")
WHERE "isOwner" = true;

WITH ranked_pending AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            PARTITION BY "orgId", LOWER(email)
            ORDER BY "createdAt" DESC, id DESC
        ) AS row_number
    FROM "OrgInvitation"
    WHERE "acceptedAt" IS NULL AND "revokedAt" IS NULL
)
UPDATE "OrgInvitation"
SET "revokedAt" = NOW()
WHERE id IN (
    SELECT id FROM ranked_pending WHERE row_number > 1
);

CREATE UNIQUE INDEX IF NOT EXISTS "OrgInvitation_one_pending_per_email"
ON "OrgInvitation" ("orgId", LOWER(email))
WHERE "acceptedAt" IS NULL AND "revokedAt" IS NULL;

WITH ranked_transfers AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            PARTITION BY "resourceType", "resourceId"
            ORDER BY "updatedAt" DESC, "createdAt" DESC, id DESC
        ) AS row_number
    FROM "TransferRequest"
    WHERE status IN ('PENDING', 'SOURCE_APPROVED', 'TARGET_APPROVED')
)
UPDATE "TransferRequest"
SET status = 'REJECTED',
    "updatedAt" = NOW()
WHERE id IN (
    SELECT id FROM ranked_transfers WHERE row_number > 1
);

CREATE UNIQUE INDEX IF NOT EXISTS "TransferRequest_one_open_per_resource"
ON "TransferRequest" ("resourceType", "resourceId")
WHERE status IN ('PENDING', 'SOURCE_APPROVED', 'TARGET_APPROVED');

ALTER TABLE "LibraryAgent"
ADD COLUMN IF NOT EXISTS "scopeKey" TEXT;

UPDATE "LibraryAgent"
SET "scopeKey" = COALESCE("organizationId", '__personal__')
    || ':' || COALESCE("teamId", '__org__')
WHERE "scopeKey" IS NULL;

ALTER TABLE "LibraryAgent"
ALTER COLUMN "scopeKey" SET DEFAULT '__scope__',
ALTER COLUMN "scopeKey" SET NOT NULL;

DROP INDEX IF EXISTS "LibraryAgent_userId_agentGraphId_agentGraphVersion_key";
CREATE UNIQUE INDEX IF NOT EXISTS "LibraryAgent_userId_agentGraphId_agentGraphVersion_scopeKey_key"
ON "LibraryAgent" ("userId", "agentGraphId", "agentGraphVersion", "scopeKey");

CREATE OR REPLACE FUNCTION sync_library_agent_scope_key()
RETURNS TRIGGER AS $$
BEGIN
    NEW."scopeKey" := COALESCE(NEW."organizationId", '__personal__')
        || ':' || COALESCE(NEW."teamId", '__org__');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql
SET search_path = pg_catalog, platform, pg_temp;

DROP TRIGGER IF EXISTS sync_library_agent_scope ON "LibraryAgent";
CREATE TRIGGER sync_library_agent_scope
BEFORE INSERT OR UPDATE OF "organizationId", "teamId"
ON "LibraryAgent" FOR EACH ROW EXECUTE FUNCTION sync_library_agent_scope_key();

CREATE OR REPLACE FUNCTION enforce_live_tenant_resource_owner()
RETURNS TRIGGER AS $$
DECLARE
    graph_lock_id TEXT;
    owner_org_id TEXT;
    owner_user_id TEXT;
    owner_team_id TEXT;
BEGIN
    IF TG_TABLE_NAME = 'AgentGraphExecution'
       AND TG_OP = 'UPDATE'
       AND to_jsonb(NEW)->>'executionStatus' IN (
           'COMPLETED', 'FAILED', 'TERMINATED'
       )
       AND to_jsonb(NEW)->>'userId'
           IS NOT DISTINCT FROM to_jsonb(OLD)->>'userId'
       AND to_jsonb(NEW)->>'organizationId'
           IS NOT DISTINCT FROM to_jsonb(OLD)->>'organizationId'
       AND to_jsonb(NEW)->>'teamId'
           IS NOT DISTINCT FROM to_jsonb(OLD)->>'teamId' THEN
        RETURN NEW;
    END IF;

    owner_org_id := COALESCE(
        to_jsonb(NEW)->>'organizationId',
        to_jsonb(NEW)->>'owningOrgId'
    );
    owner_team_id := COALESCE(
        to_jsonb(NEW)->>'teamIdRestriction',
        to_jsonb(NEW)->>'teamId'
    );
    IF owner_org_id IS NULL AND owner_team_id IS NOT NULL THEN
        RAISE EXCEPTION 'team-scoped resource requires organization tenancy'
            USING ERRCODE = '23514';
    END IF;
    IF owner_org_id IS NULL THEN
        RETURN NEW;
    END IF;

    owner_user_id := COALESCE(
        to_jsonb(NEW)->>'userId',
        to_jsonb(NEW)->>'ownerUserId',
        to_jsonb(NEW)->>'owningUserId'
    );
    IF owner_user_id IS NULL THEN
        RETURN NEW;
    END IF;

    PERFORM 1 FROM "User" WHERE id = owner_user_id FOR UPDATE;
    PERFORM 1 FROM "Organization"
    WHERE id = owner_org_id FOR UPDATE;
    IF NOT EXISTS (
        SELECT 1
        FROM "OrgMember" member
        JOIN "Organization" org ON org.id = member."orgId"
        WHERE member."orgId" = owner_org_id
          AND member."userId" = owner_user_id
          AND member.status = 'ACTIVE'
          AND org."deletedAt" IS NULL
    ) THEN
        RAISE EXCEPTION 'resource owner lacks an active organization membership'
            USING ERRCODE = '23514';
    END IF;

    IF owner_team_id IS NOT NULL THEN
        PERFORM 1 FROM "Team"
        WHERE id = owner_team_id
          AND "orgId" = owner_org_id
        FOR UPDATE;
    END IF;
    IF owner_team_id IS NOT NULL AND NOT EXISTS (
        SELECT 1
        FROM "TeamMember" member
        JOIN "Team" team ON team.id = member."teamId"
        WHERE member."teamId" = owner_team_id
          AND member."userId" = owner_user_id
          AND member.status = 'ACTIVE'
          AND team."orgId" = owner_org_id
          AND team."archivedAt" IS NULL
    ) THEN
        RAISE EXCEPTION 'resource owner lacks an active workspace membership'
            USING ERRCODE = '23514';
    END IF;

    graph_lock_id := COALESCE(
        to_jsonb(NEW)->>'agentGraphId',
        CASE WHEN TG_TABLE_NAME = 'AgentGraph' THEN to_jsonb(NEW)->>'id' END
    );
    IF graph_lock_id IS NOT NULL THEN
        PERFORM 1 FROM "AgentGraph" WHERE id = graph_lock_id FOR UPDATE;
    END IF;
    IF TG_TABLE_NAME = 'StoreListing' AND NOT EXISTS (
        SELECT 1 FROM "AgentGraph"
        WHERE id = graph_lock_id
          AND "organizationId" = owner_org_id
    ) THEN
        RAISE EXCEPTION 'store listing tenancy must match its graph'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql
SET search_path = pg_catalog, platform, pg_temp;

CREATE OR REPLACE FUNCTION lock_org_member_tenancy_change()
RETURNS TRIGGER AS $$
DECLARE
    member_org_id TEXT;
    member_user_id TEXT;
BEGIN
    IF TG_OP = 'DELETE' THEN
        member_org_id := OLD."orgId";
        member_user_id := OLD."userId";
    ELSE
        member_org_id := NEW."orgId";
        member_user_id := NEW."userId";
    END IF;
    PERFORM pg_advisory_xact_lock(
        hashtextextended(
            'tenancy:org-user:' || member_org_id || ':' || member_user_id,
            0
        )
    );
    PERFORM pg_advisory_xact_lock(
        hashtextextended('tenancy:org:' || member_org_id, 0)
    );
    PERFORM 1 FROM "User" WHERE id = member_user_id FOR UPDATE;
    PERFORM 1 FROM "Organization" WHERE id = member_org_id FOR UPDATE;
    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql
SET search_path = pg_catalog, platform, pg_temp;

CREATE OR REPLACE FUNCTION lock_team_member_tenancy_change()
RETURNS TRIGGER AS $$
DECLARE
    member_org_id TEXT;
    member_team_id TEXT;
    member_user_id TEXT;
BEGIN
    IF TG_OP = 'DELETE' THEN
        member_team_id := OLD."teamId";
        member_user_id := OLD."userId";
    ELSE
        member_team_id := NEW."teamId";
        member_user_id := NEW."userId";
    END IF;
    SELECT "orgId" INTO member_org_id FROM "Team" WHERE id = member_team_id;
    PERFORM pg_advisory_xact_lock(
        hashtextextended(
            'tenancy:org-user:' || member_org_id || ':' || member_user_id,
            0
        )
    );
    PERFORM pg_advisory_xact_lock(
        hashtextextended('tenancy:org:' || member_org_id, 0)
    );
    PERFORM pg_advisory_xact_lock(
        hashtextextended('tenancy:team:' || member_team_id, 0)
    );
    PERFORM 1 FROM "User" WHERE id = member_user_id FOR UPDATE;
    PERFORM 1 FROM "Organization" WHERE id = member_org_id FOR UPDATE;
    PERFORM 1 FROM "Team" WHERE id = member_team_id FOR UPDATE;
    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql
SET search_path = pg_catalog, platform, pg_temp;

CREATE OR REPLACE FUNCTION enforce_store_listing_version_tenancy()
RETURNS TRIGGER AS $$
DECLARE
    graph_org_id TEXT;
    listing_org_id TEXT;
BEGIN
    SELECT "organizationId" INTO graph_org_id
    FROM "AgentGraph"
    WHERE id = NEW."agentGraphId" AND version = NEW."agentGraphVersion"
    FOR UPDATE;
    SELECT "owningOrgId" INTO listing_org_id
    FROM "StoreListing"
    WHERE id = NEW."storeListingId";
    IF NEW."organizationId" IS DISTINCT FROM graph_org_id
       OR NEW."organizationId" IS DISTINCT FROM listing_org_id THEN
        RAISE EXCEPTION 'listing version tenancy must match its graph and listing'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql
SET search_path = pg_catalog, platform, pg_temp;

CREATE OR REPLACE FUNCTION enforce_agent_graph_grant_tenancy()
RETURNS TRIGGER AS $$
DECLARE
    graph_org_id TEXT;
    principal_org_id TEXT;
BEGIN
    SELECT "organizationId" INTO graph_org_id
    FROM "AgentGraph"
    WHERE id = NEW."agentGraphId"
    ORDER BY version DESC
    LIMIT 1
    FOR UPDATE;
    IF NEW."principalType" = 'TEAM' THEN
        SELECT "orgId" INTO principal_org_id
        FROM "Team"
        WHERE id = NEW."principalId" AND "archivedAt" IS NULL
        FOR UPDATE;
    END IF;
    IF graph_org_id IS DISTINCT FROM NEW."organizationId"
       OR (NEW."principalType" = 'TEAM'
           AND principal_org_id IS DISTINCT FROM NEW."organizationId") THEN
        RAISE EXCEPTION 'grant tenancy must match its graph and principal'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql
SET search_path = pg_catalog, platform, pg_temp;

CREATE OR REPLACE FUNCTION enforce_owned_library_agent_tenancy()
RETURNS TRIGGER AS $$
DECLARE
    graph_user_id TEXT;
    graph_org_id TEXT;
    graph_team_id TEXT;
BEGIN
    IF NOT NEW."isCreatedByUser" THEN
        RETURN NEW;
    END IF;
    SELECT "userId", "organizationId", "teamId"
    INTO graph_user_id, graph_org_id, graph_team_id
    FROM "AgentGraph"
    WHERE id = NEW."agentGraphId" AND version = NEW."agentGraphVersion"
    FOR UPDATE;
    IF graph_user_id IS DISTINCT FROM NEW."userId"
       OR graph_org_id IS DISTINCT FROM NEW."organizationId"
       OR graph_team_id IS DISTINCT FROM NEW."teamId" THEN
        RAISE EXCEPTION 'owned library entry tenancy must match its graph'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql
SET search_path = pg_catalog, platform, pg_temp;

CREATE OR REPLACE FUNCTION lock_expert_workflow_graph()
RETURNS TRIGGER AS $$
DECLARE
    graph_lock_id TEXT;
BEGIN
    IF NEW."libraryAgentId" IS NULL THEN
        RETURN NEW;
    END IF;
    SELECT "agentGraphId" INTO graph_lock_id
    FROM "LibraryAgent"
    WHERE id = NEW."libraryAgentId"
    FOR UPDATE;
    IF graph_lock_id IS NOT NULL THEN
        PERFORM 1 FROM "AgentGraph" WHERE id = graph_lock_id FOR UPDATE;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql
SET search_path = pg_catalog, platform, pg_temp;

CREATE OR REPLACE FUNCTION enforce_live_team_member_owner()
RETURNS TRIGGER AS $$
DECLARE
    owner_org_id TEXT;
BEGIN
    IF NEW.status <> 'ACTIVE' THEN
        RETURN NEW;
    END IF;

    PERFORM 1 FROM "User" WHERE id = NEW."userId" FOR UPDATE;
    SELECT "orgId" INTO owner_org_id FROM "Team" WHERE id = NEW."teamId";
    PERFORM 1 FROM "Organization" WHERE id = owner_org_id FOR UPDATE;
    PERFORM 1 FROM "Team"
    WHERE id = NEW."teamId" AND "orgId" = owner_org_id FOR UPDATE;
    IF owner_org_id IS NULL OR NOT EXISTS (
        SELECT 1
        FROM "OrgMember" member
        JOIN "Organization" org ON org.id = member."orgId"
        WHERE member."orgId" = owner_org_id
          AND member."userId" = NEW."userId"
          AND member.status = 'ACTIVE'
          AND org."deletedAt" IS NULL
    ) THEN
        RAISE EXCEPTION 'workspace member lacks an active organization membership'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql
SET search_path = pg_catalog, platform, pg_temp;

DO $$
DECLARE
    table_name TEXT;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'AgentGraph',
        'AgentGraphExecution',
        'LibraryAgent',
        'LibraryFolder',
        'AgentPreset',
        'IntegrationWebhook',
        'ChatSession'
    ] LOOP
        EXECUTE format(
            'DROP TRIGGER IF EXISTS enforce_live_tenant_owner ON %I',
            table_name
        );
        EXECUTE format(
            'CREATE TRIGGER enforce_live_tenant_owner '
            'BEFORE INSERT OR UPDATE OF "userId", "organizationId", "teamId" '
            'ON %I FOR EACH ROW EXECUTE FUNCTION enforce_live_tenant_resource_owner()',
            table_name
        );
    END LOOP;
END;
$$;

DROP TRIGGER IF EXISTS enforce_live_tenant_owner ON "AgentGraphExecution";
CREATE TRIGGER enforce_live_tenant_owner
BEFORE INSERT OR UPDATE OF "userId", "organizationId", "teamId", "executionStatus"
ON "AgentGraphExecution" FOR EACH ROW EXECUTE FUNCTION enforce_live_tenant_resource_owner();

DROP TRIGGER IF EXISTS a_lock_live_tenancy_change ON "OrgMember";
CREATE TRIGGER a_lock_live_tenancy_change
BEFORE INSERT OR UPDATE OR DELETE
ON "OrgMember" FOR EACH ROW EXECUTE FUNCTION lock_org_member_tenancy_change();

DROP TRIGGER IF EXISTS a_lock_live_tenancy_change ON "TeamMember";
CREATE TRIGGER a_lock_live_tenancy_change
BEFORE INSERT OR UPDATE OR DELETE
ON "TeamMember" FOR EACH ROW EXECUTE FUNCTION lock_team_member_tenancy_change();

DROP TRIGGER IF EXISTS enforce_live_tenant_owner ON "Expert";
CREATE TRIGGER enforce_live_tenant_owner
BEFORE INSERT OR UPDATE OF "ownerUserId", "organizationId", "teamId"
ON "Expert" FOR EACH ROW EXECUTE FUNCTION enforce_live_tenant_resource_owner();

DROP TRIGGER IF EXISTS enforce_live_tenant_owner ON "StoreListing";
CREATE TRIGGER enforce_live_tenant_owner
BEFORE INSERT OR UPDATE OF "owningUserId", "owningOrgId"
ON "StoreListing" FOR EACH ROW EXECUTE FUNCTION enforce_live_tenant_resource_owner();

DROP TRIGGER IF EXISTS enforce_listing_version_tenancy ON "StoreListingVersion";
CREATE TRIGGER enforce_listing_version_tenancy
BEFORE INSERT OR UPDATE OF "agentGraphId", "agentGraphVersion", "storeListingId", "organizationId"
ON "StoreListingVersion" FOR EACH ROW EXECUTE FUNCTION enforce_store_listing_version_tenancy();

DROP TRIGGER IF EXISTS enforce_graph_grant_tenancy ON "AgentGraphGrant";
CREATE TRIGGER enforce_graph_grant_tenancy
BEFORE INSERT OR UPDATE OF "agentGraphId", "principalType", "principalId", "organizationId"
ON "AgentGraphGrant" FOR EACH ROW EXECUTE FUNCTION enforce_agent_graph_grant_tenancy();

DROP TRIGGER IF EXISTS enforce_owned_library_agent_tenancy ON "LibraryAgent";
CREATE TRIGGER enforce_owned_library_agent_tenancy
BEFORE INSERT OR UPDATE OF "userId", "agentGraphId", "agentGraphVersion", "organizationId", "teamId", "isCreatedByUser"
ON "LibraryAgent" FOR EACH ROW EXECUTE FUNCTION enforce_owned_library_agent_tenancy();

DROP TRIGGER IF EXISTS lock_agent_graph ON "ExpertWorkflow";
CREATE TRIGGER lock_agent_graph
BEFORE INSERT OR UPDATE OF "libraryAgentId"
ON "ExpertWorkflow" FOR EACH ROW EXECUTE FUNCTION lock_expert_workflow_graph();

DROP TRIGGER IF EXISTS enforce_live_tenant_owner ON "APIKey";
CREATE TRIGGER enforce_live_tenant_owner
BEFORE INSERT OR UPDATE OF "userId", "organizationId", "teamId", "teamIdRestriction"
ON "APIKey" FOR EACH ROW EXECUTE FUNCTION enforce_live_tenant_resource_owner();

DROP TRIGGER IF EXISTS enforce_live_org_owner ON "TeamMember";
CREATE TRIGGER enforce_live_org_owner
BEFORE INSERT OR UPDATE OF "userId", "teamId", status
ON "TeamMember" FOR EACH ROW EXECUTE FUNCTION enforce_live_team_member_owner();
