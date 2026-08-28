DROP INDEX IF EXISTS "UserWorkspaceFolder_workspaceId_name_root_key";

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

    PERFORM 1 FROM "User" WHERE id = owner_user_id FOR SHARE;
    PERFORM 1 FROM "Organization"
    WHERE id = owner_org_id FOR SHARE;
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
        FOR SHARE;
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
        PERFORM 1 FROM "AgentGraph" WHERE id = graph_lock_id FOR SHARE;
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
