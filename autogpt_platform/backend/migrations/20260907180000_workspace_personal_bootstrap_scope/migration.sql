CREATE OR REPLACE FUNCTION workspace_personal_bootstrap_allowed(
    workspace_id TEXT, new_org_id TEXT, new_team_id TEXT
)
RETURNS BOOLEAN AS $$
DECLARE
    owner_user_id TEXT;
BEGIN
    IF new_org_id IS NULL OR new_team_id IS NULL THEN
        RETURN false;
    END IF;
    SELECT "userId" INTO owner_user_id
    FROM "UserWorkspace" WHERE id = workspace_id FOR KEY SHARE;
    IF owner_user_id IS NULL THEN
        RETURN false;
    END IF;
    PERFORM pg_advisory_xact_lock_shared(hashtextextended(
        'tenancy:org-user:' || new_org_id || ':' || owner_user_id, 0
    ));
    PERFORM pg_advisory_xact_lock_shared(hashtextextended('tenancy:org:' || new_org_id, 0));
    PERFORM pg_advisory_xact_lock_shared(hashtextextended('tenancy:team:' || new_team_id, 0));
    RETURN EXISTS (
        SELECT 1 FROM "Organization" org
        JOIN "OrgMember" member ON member."orgId" = org.id
        JOIN "Team" team ON team."orgId" = org.id
        JOIN "TeamMember" team_member ON team_member."teamId" = team.id
        WHERE org.id = new_org_id AND org."isPersonal" AND org."deletedAt" IS NULL
          AND member."userId" = owner_user_id AND member."isOwner" AND member.status = 'ACTIVE'
          AND team.id = new_team_id AND team."isDefault" AND team."archivedAt" IS NULL
          AND team_member."userId" = owner_user_id AND team_member.status = 'ACTIVE'
    );
END;
$$ LANGUAGE plpgsql
SET search_path = pg_catalog, platform, pg_temp;

CREATE OR REPLACE FUNCTION enforce_workspace_artifact_scope()
RETURNS TRIGGER AS $$
DECLARE
    owner_user_id TEXT;
    source_user_id TEXT;
    source_org_id TEXT;
    source_team_id TEXT;
    folder_workspace_id TEXT;
    folder_org_id TEXT;
    folder_team_id TEXT;
    folder_scope_resolved BOOLEAN;
BEGIN
    IF TG_OP = 'UPDATE' AND (
        NEW."workspaceId" IS DISTINCT FROM OLD."workspaceId"
        OR NEW."organizationId" IS DISTINCT FROM OLD."organizationId"
        OR NEW."teamId" IS DISTINCT FROM OLD."teamId"
        OR NEW."sessionId" IS DISTINCT FROM OLD."sessionId"
        OR NEW."executionId" IS DISTINCT FROM OLD."executionId"
        OR NEW."scopeResolved" IS DISTINCT FROM OLD."scopeResolved"
        OR NEW."isUserGlobalConfig" IS DISTINCT FROM OLD."isUserGlobalConfig"
    ) AND NOT (
        OLD."organizationId" IS NULL AND OLD."teamId" IS NULL
        AND OLD."scopeResolved" AND NEW."scopeResolved"
        AND OLD."scopeKey" = '__personal__:__org__'
        AND NEW."workspaceId" IS NOT DISTINCT FROM OLD."workspaceId"
        AND workspace_personal_bootstrap_allowed(
            NEW."workspaceId", NEW."organizationId", NEW."teamId"
        )
        AND NOT OLD."isUserGlobalConfig" AND NOT NEW."isUserGlobalConfig"
        AND NEW."sessionId" IS NOT DISTINCT FROM OLD."sessionId"
        AND NEW."executionId" IS NOT DISTINCT FROM OLD."executionId"
        AND NEW."folderId" IS NOT DISTINCT FROM OLD."folderId"
        AND (NEW."sessionId" IS NOT NULL OR NEW."executionId" IS NOT NULL)
    ) THEN
        RAISE EXCEPTION 'workspace artifact scope is immutable'
            USING ERRCODE = '23514';
    END IF;

    IF NOT NEW."scopeResolved" THEN
        IF TG_OP = 'INSERT' THEN
            RAISE EXCEPTION 'new workspace artifacts require resolved scope'
                USING ERRCODE = '23514';
        END IF;
        NEW."scopeKey" := '__legacy_quarantine__';
        RETURN NEW;
    END IF;

    IF NEW."isUserGlobalConfig" THEN
        IF NEW."organizationId" IS NOT NULL
           OR NEW."teamId" IS NOT NULL
           OR NEW."sessionId" IS NOT NULL
           OR NEW."executionId" IS NOT NULL
           OR NEW."folderId" IS NOT NULL THEN
            RAISE EXCEPTION 'user-global config cannot carry tenant scope'
                USING ERRCODE = '23514';
        END IF;
        NEW."scopeKey" := '__user_global__';
        RETURN NEW;
    END IF;

    IF NEW."teamId" IS NOT NULL AND NEW."organizationId" IS NULL THEN
        RAISE EXCEPTION 'workspace artifact team requires organization'
            USING ERRCODE = '23514';
    END IF;
    IF NEW."sessionId" IS NOT NULL AND NEW."executionId" IS NOT NULL THEN
        RAISE EXCEPTION 'workspace artifact cannot have two sources'
            USING ERRCODE = '23514';
    END IF;

    SELECT "userId" INTO owner_user_id
    FROM "UserWorkspace"
    WHERE id = NEW."workspaceId"
    FOR KEY SHARE;
    IF owner_user_id IS NULL THEN
        RAISE EXCEPTION 'workspace artifact requires an owning workspace'
            USING ERRCODE = '23503';
    END IF;

    IF NEW."organizationId" IS NOT NULL THEN
        PERFORM pg_advisory_xact_lock_shared(hashtextextended(
            'tenancy:org-user:' || NEW."organizationId" || ':' || owner_user_id, 0
        ));
        PERFORM pg_advisory_xact_lock_shared(hashtextextended(
            'tenancy:org:' || NEW."organizationId", 0
        ));
        IF NEW."teamId" IS NOT NULL THEN
            PERFORM pg_advisory_xact_lock_shared(hashtextextended(
                'tenancy:team:' || NEW."teamId", 0
            ));
        END IF;
        IF NOT EXISTS (
            SELECT 1
            FROM "OrgMember" member
            JOIN "Organization" org ON org.id = member."orgId"
            WHERE member."orgId" = NEW."organizationId"
              AND member."userId" = owner_user_id
              AND member.status = 'ACTIVE'
              AND org."deletedAt" IS NULL
        ) THEN
            RAISE EXCEPTION 'workspace artifact owner lacks live organization access'
                USING ERRCODE = '23514';
        END IF;
        IF NEW."teamId" IS NOT NULL AND NOT EXISTS (
            SELECT 1
            FROM "TeamMember" member
            JOIN "Team" team ON team.id = member."teamId"
            WHERE member."teamId" = NEW."teamId"
              AND member."userId" = owner_user_id
              AND member.status = 'ACTIVE'
              AND team."orgId" = NEW."organizationId"
              AND team."archivedAt" IS NULL
        ) THEN
            RAISE EXCEPTION 'workspace artifact owner lacks live workspace access'
                USING ERRCODE = '23514';
        END IF;
    END IF;

    IF NEW."sessionId" IS NOT NULL THEN
        SELECT "userId", "organizationId", "teamId"
        INTO source_user_id, source_org_id, source_team_id
        FROM "ChatSession"
        WHERE id = NEW."sessionId"
        FOR KEY SHARE;
        IF (source_user_id, source_org_id, source_team_id) IS DISTINCT FROM
           (owner_user_id, NEW."organizationId", NEW."teamId") THEN
            RAISE EXCEPTION 'workspace artifact session scope mismatch'
                USING ERRCODE = '23514';
        END IF;
    END IF;

    IF NEW."executionId" IS NOT NULL THEN
        SELECT "userId", "organizationId", "teamId"
        INTO source_user_id, source_org_id, source_team_id
        FROM "AgentGraphExecution"
        WHERE id = NEW."executionId"
        FOR KEY SHARE;
        IF (source_user_id, source_org_id, source_team_id) IS DISTINCT FROM
           (owner_user_id, NEW."organizationId", NEW."teamId") THEN
            RAISE EXCEPTION 'workspace artifact execution scope mismatch'
                USING ERRCODE = '23514';
        END IF;
    END IF;

    IF NEW."folderId" IS NOT NULL THEN
        SELECT "workspaceId", "organizationId", "teamId", "scopeResolved"
        INTO folder_workspace_id, folder_org_id, folder_team_id, folder_scope_resolved
        FROM "UserWorkspaceFolder"
        WHERE id = NEW."folderId" AND "isDeleted" = false
        FOR KEY SHARE;
        IF NOT COALESCE(folder_scope_resolved, false)
           OR (folder_workspace_id, folder_org_id, folder_team_id) IS DISTINCT FROM
              (NEW."workspaceId", NEW."organizationId", NEW."teamId") THEN
            RAISE EXCEPTION 'workspace artifact folder scope mismatch'
                USING ERRCODE = '23514';
        END IF;
    END IF;

    NEW."scopeKey" := COALESCE(NEW."organizationId", '__personal__')
        || ':' || COALESCE(NEW."teamId", '__org__');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql
SET search_path = pg_catalog, platform, pg_temp;

CREATE OR REPLACE FUNCTION enforce_workspace_folder_scope()
RETURNS TRIGGER AS $$
DECLARE
    owner_user_id TEXT;
    parent_workspace_id TEXT;
    parent_org_id TEXT;
    parent_team_id TEXT;
    parent_scope_resolved BOOLEAN;
BEGIN
    IF TG_OP = 'UPDATE' AND (
        NEW."workspaceId" IS DISTINCT FROM OLD."workspaceId"
        OR NEW."organizationId" IS DISTINCT FROM OLD."organizationId"
        OR NEW."teamId" IS DISTINCT FROM OLD."teamId"
        OR NEW."scopeResolved" IS DISTINCT FROM OLD."scopeResolved"
    ) AND NOT (
        OLD."organizationId" IS NULL AND OLD."teamId" IS NULL
        AND OLD."scopeResolved" AND NEW."scopeResolved"
        AND OLD."scopeKey" = '__personal__:__org__'
        AND NEW."workspaceId" IS NOT DISTINCT FROM OLD."workspaceId"
        AND workspace_personal_bootstrap_allowed(
            NEW."workspaceId", NEW."organizationId", NEW."teamId"
        )
        AND (
            NEW."parentId" IS NOT DISTINCT FROM OLD."parentId" OR (
                NEW."parentId" IS NULL AND EXISTS (
                    SELECT 1 FROM "UserWorkspaceFolder" parent
                    WHERE parent.id = OLD."parentId"
                      AND parent."workspaceId" = OLD."workspaceId"
                      AND NOT parent."scopeResolved"
                )
            )
        )
    ) THEN
        RAISE EXCEPTION 'workspace folder scope is immutable'
            USING ERRCODE = '23514';
    END IF;
    IF NOT NEW."scopeResolved" THEN
        IF TG_OP = 'INSERT' THEN
            RAISE EXCEPTION 'new workspace folders require resolved scope'
                USING ERRCODE = '23514';
        END IF;
        NEW."scopeKey" := '__legacy_quarantine__';
        RETURN NEW;
    END IF;
    IF NEW."teamId" IS NOT NULL AND NEW."organizationId" IS NULL THEN
        RAISE EXCEPTION 'workspace folder team requires organization'
            USING ERRCODE = '23514';
    END IF;
    SELECT "userId" INTO owner_user_id
    FROM "UserWorkspace"
    WHERE id = NEW."workspaceId"
    FOR KEY SHARE;
    IF owner_user_id IS NULL THEN
        RAISE EXCEPTION 'workspace folder requires an owning workspace'
            USING ERRCODE = '23503';
    END IF;
    IF NEW."organizationId" IS NOT NULL THEN
        PERFORM pg_advisory_xact_lock_shared(hashtextextended(
            'tenancy:org-user:' || NEW."organizationId" || ':' || owner_user_id, 0
        ));
        PERFORM pg_advisory_xact_lock_shared(hashtextextended(
            'tenancy:org:' || NEW."organizationId", 0
        ));
        IF NEW."teamId" IS NOT NULL THEN
            PERFORM pg_advisory_xact_lock_shared(hashtextextended(
                'tenancy:team:' || NEW."teamId", 0
            ));
        END IF;
        IF NOT EXISTS (
            SELECT 1
            FROM "OrgMember" member
            JOIN "Organization" org ON org.id = member."orgId"
            WHERE member."orgId" = NEW."organizationId"
              AND member."userId" = owner_user_id
              AND member.status = 'ACTIVE'
              AND org."deletedAt" IS NULL
        ) THEN
            RAISE EXCEPTION 'workspace folder owner lacks live organization access'
                USING ERRCODE = '23514';
        END IF;
        IF NEW."teamId" IS NOT NULL AND NOT EXISTS (
            SELECT 1
            FROM "TeamMember" member
            JOIN "Team" team ON team.id = member."teamId"
            WHERE member."teamId" = NEW."teamId"
              AND member."userId" = owner_user_id
              AND member.status = 'ACTIVE'
              AND team."orgId" = NEW."organizationId"
              AND team."archivedAt" IS NULL
        ) THEN
            RAISE EXCEPTION 'workspace folder owner lacks live workspace access'
                USING ERRCODE = '23514';
        END IF;
    END IF;
    IF NEW."parentId" IS NOT NULL THEN
        SELECT "workspaceId", "organizationId", "teamId", "scopeResolved"
        INTO parent_workspace_id, parent_org_id, parent_team_id, parent_scope_resolved
        FROM "UserWorkspaceFolder"
        WHERE id = NEW."parentId" AND "isDeleted" = false
        FOR KEY SHARE;
        IF NOT COALESCE(parent_scope_resolved, false)
           OR (parent_workspace_id, parent_org_id, parent_team_id) IS DISTINCT FROM
              (NEW."workspaceId", NEW."organizationId", NEW."teamId") THEN
            RAISE EXCEPTION 'workspace folder parent scope mismatch'
                USING ERRCODE = '23514';
        END IF;
    END IF;
    NEW."scopeKey" := COALESCE(NEW."organizationId", '__personal__')
        || ':' || COALESCE(NEW."teamId", '__org__');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql
SET search_path = pg_catalog, platform, pg_temp;

DO $$
DECLARE
    app_schema TEXT := current_schema();
BEGIN
    EXECUTE format(
        'ALTER FUNCTION %I.workspace_personal_bootstrap_allowed(TEXT, TEXT, TEXT) SET search_path = pg_catalog, %I, pg_temp',
        app_schema, app_schema
    );
    EXECUTE format(
        'ALTER FUNCTION %I.enforce_workspace_artifact_scope() SET search_path = pg_catalog, %I, pg_temp',
        app_schema, app_schema
    );
    EXECUTE format(
        'ALTER FUNCTION %I.enforce_workspace_folder_scope() SET search_path = pg_catalog, %I, pg_temp',
        app_schema, app_schema
    );
END $$;
