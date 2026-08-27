ALTER TABLE "UserWorkspaceFile"
ADD COLUMN "organizationId" TEXT,
ADD COLUMN "teamId" TEXT,
ADD COLUMN "sessionId" TEXT,
ADD COLUMN "executionId" TEXT,
ADD COLUMN "scopeKey" TEXT NOT NULL DEFAULT '__legacy_quarantine__',
ADD COLUMN "scopeResolved" BOOLEAN NOT NULL DEFAULT false,
ADD COLUMN "isUserGlobalConfig" BOOLEAN NOT NULL DEFAULT false;

ALTER TABLE "UserWorkspaceFolder"
ADD COLUMN "organizationId" TEXT,
ADD COLUMN "teamId" TEXT,
ADD COLUMN "scopeKey" TEXT NOT NULL DEFAULT '__legacy_quarantine__',
ADD COLUMN "scopeResolved" BOOLEAN NOT NULL DEFAULT false;

UPDATE "UserWorkspaceFile"
SET "scopeResolved" = true,
    "isUserGlobalConfig" = true,
    "scopeKey" = '__user_global__'
WHERE path LIKE '/skills/%';

WITH session_files AS (
    SELECT
        file.id,
        session.id AS session_id,
        session."organizationId" AS organization_id,
        session."teamId" AS team_id
    FROM "UserWorkspaceFile" file
    JOIN "UserWorkspace" workspace ON workspace.id = file."workspaceId"
    JOIN "ChatSession" session
      ON session.id = substring(file.path FROM '^/sessions/([^/]+)/')
     AND session."userId" = workspace."userId"
    WHERE file.path ~ '^/sessions/[^/]+/'
      AND file."isUserGlobalConfig" = false
)
UPDATE "UserWorkspaceFile" file
SET "organizationId" = session_files.organization_id,
    "teamId" = session_files.team_id,
    "sessionId" = session_files.session_id,
    "scopeResolved" = true,
    "scopeKey" = COALESCE(session_files.organization_id, '__personal__')
        || ':' || COALESCE(session_files.team_id, '__org__')
FROM session_files
WHERE file.id = session_files.id;

WITH compatible_folders AS (
    SELECT
        "folderId",
        MIN("organizationId") AS organization_id,
        MIN("teamId") AS team_id,
        MIN("scopeKey") AS scope_key
    FROM "UserWorkspaceFile"
    WHERE "folderId" IS NOT NULL
    GROUP BY "folderId"
    HAVING BOOL_AND("scopeResolved")
       AND COUNT(DISTINCT "scopeKey") = 1
)
UPDATE "UserWorkspaceFolder" folder
SET "organizationId" = compatible_folders.organization_id,
    "teamId" = compatible_folders.team_id,
    "scopeKey" = compatible_folders.scope_key,
    "scopeResolved" = true
FROM compatible_folders
WHERE folder.id = compatible_folders."folderId";

UPDATE "UserWorkspaceFile" file
SET "folderId" = NULL
FROM "UserWorkspaceFolder" folder
WHERE file."folderId" = folder.id
  AND folder."scopeResolved" = false;

ALTER TABLE "UserWorkspaceFile"
ALTER COLUMN "scopeKey" SET DEFAULT '__scope__',
ALTER COLUMN "scopeResolved" SET DEFAULT true;

ALTER TABLE "UserWorkspaceFolder"
ALTER COLUMN "scopeKey" SET DEFAULT '__scope__',
ALTER COLUMN "scopeResolved" SET DEFAULT true;

DROP INDEX IF EXISTS "UserWorkspaceFile_workspaceId_path_key";
CREATE UNIQUE INDEX "UserWorkspaceFile_workspaceId_scopeKey_path_key"
ON "UserWorkspaceFile"("workspaceId", "scopeKey", path);

DROP INDEX IF EXISTS "UserWorkspaceFolder_workspaceId_parentId_name_key";
CREATE UNIQUE INDEX "UserWorkspaceFolder_workspaceId_scopeKey_parentId_name_key"
ON "UserWorkspaceFolder"("workspaceId", "scopeKey", "parentId", name);
CREATE UNIQUE INDEX "UserWorkspaceFolder_workspace_scope_root_name_key"
ON "UserWorkspaceFolder"("workspaceId", "scopeKey", name)
WHERE "parentId" IS NULL AND "isDeleted" = false;

CREATE INDEX "UserWorkspaceFile_workspaceId_organizationId_teamId_isDeleted_idx"
ON "UserWorkspaceFile"("workspaceId", "organizationId", "teamId", "isDeleted");
CREATE INDEX "UserWorkspaceFile_sessionId_idx" ON "UserWorkspaceFile"("sessionId");
CREATE INDEX "UserWorkspaceFile_executionId_idx" ON "UserWorkspaceFile"("executionId");
CREATE INDEX "UserWorkspaceFolder_workspaceId_organizationId_teamId_isDeleted_idx"
ON "UserWorkspaceFolder"("workspaceId", "organizationId", "teamId", "isDeleted");

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

CREATE OR REPLACE FUNCTION enforce_shared_workspace_file_scope()
RETURNS TRIGGER AS $$
DECLARE
    parent_user_id TEXT;
    parent_org_id TEXT;
    parent_team_id TEXT;
    file_user_id TEXT;
    file_org_id TEXT;
    file_team_id TEXT;
    file_scope_resolved BOOLEAN;
BEGIN
    IF TG_TABLE_NAME = 'SharedChatFile' THEN
        SELECT "userId", "organizationId", "teamId"
        INTO parent_user_id, parent_org_id, parent_team_id
        FROM "ChatSession" WHERE id = NEW."sessionId" FOR KEY SHARE;
    ELSE
        SELECT "userId", "organizationId", "teamId"
        INTO parent_user_id, parent_org_id, parent_team_id
        FROM "AgentGraphExecution" WHERE id = NEW."executionId" FOR KEY SHARE;
    END IF;
    SELECT workspace."userId", file."organizationId", file."teamId", file."scopeResolved"
    INTO file_user_id, file_org_id, file_team_id, file_scope_resolved
    FROM "UserWorkspaceFile" file
    JOIN "UserWorkspace" workspace ON workspace.id = file."workspaceId"
    WHERE file.id = NEW."fileId" AND file."isDeleted" = false
    FOR KEY SHARE OF file;
    IF NOT COALESCE(file_scope_resolved, false)
       OR (file_user_id, file_org_id, file_team_id) IS DISTINCT FROM
          (parent_user_id, parent_org_id, parent_team_id) THEN
        RAISE EXCEPTION 'shared workspace file scope mismatch'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql
SET search_path = pg_catalog, platform, pg_temp;

DROP TRIGGER IF EXISTS enforce_workspace_artifact_scope ON "UserWorkspaceFile";
CREATE TRIGGER enforce_workspace_artifact_scope
BEFORE INSERT OR UPDATE OF "workspaceId", "organizationId", "teamId", "sessionId", "executionId", "scopeKey", "scopeResolved", "isUserGlobalConfig", "folderId"
ON "UserWorkspaceFile" FOR EACH ROW EXECUTE FUNCTION enforce_workspace_artifact_scope();

DROP TRIGGER IF EXISTS enforce_workspace_folder_scope ON "UserWorkspaceFolder";
CREATE TRIGGER enforce_workspace_folder_scope
BEFORE INSERT OR UPDATE OF "workspaceId", "organizationId", "teamId", "scopeKey", "scopeResolved", "parentId"
ON "UserWorkspaceFolder" FOR EACH ROW EXECUTE FUNCTION enforce_workspace_folder_scope();

DROP TRIGGER IF EXISTS enforce_shared_workspace_file_scope ON "SharedChatFile";
CREATE TRIGGER enforce_shared_workspace_file_scope
BEFORE INSERT OR UPDATE OF "sessionId", "fileId"
ON "SharedChatFile" FOR EACH ROW EXECUTE FUNCTION enforce_shared_workspace_file_scope();

DROP TRIGGER IF EXISTS enforce_shared_workspace_file_scope ON "SharedExecutionFile";
CREATE TRIGGER enforce_shared_workspace_file_scope
BEFORE INSERT OR UPDATE OF "executionId", "fileId"
ON "SharedExecutionFile" FOR EACH ROW EXECUTE FUNCTION enforce_shared_workspace_file_scope();
