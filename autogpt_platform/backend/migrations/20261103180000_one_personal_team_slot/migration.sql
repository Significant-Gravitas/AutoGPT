-- One slot for personal work: a personal organization's default team.
--
-- A personal organization's rows sat in two places: at its org-home (teamId
-- NULL), where the web app saved them, and in its default team, where the
-- startup backfill and every internal writer put them. Reads restrict to the
-- default team, so org-home rows went missing. This moves them, and a trigger
-- sends every later write to the default team as well.
--
-- Trigger order: PostgreSQL fires same-event row triggers in name order (C
-- collation) and hands each the row the previous one returned, so
-- a0_normalize_personal_team runs before a_lock_*, enforce_* and
-- sync_library_agent_scope, which then check and key the normalized row.
-- Uppercase names sort first: "AlertCondition_execution_scope_guard" still
-- runs before it, so an alert must name its source execution's exact scope,
-- which the alert writer already does.
--
-- Idempotent: functions are replaced, triggers recreated, and every move only
-- touches rows still at org-home.
BEGIN;
SET LOCAL statement_timeout = 0;

CREATE OR REPLACE FUNCTION personal_default_team_id(org_id TEXT)
RETURNS TEXT AS $$
    SELECT team.id
    FROM "Organization" AS org
    JOIN "Team" AS team
      ON team."orgId" = org.id
     AND team."isDefault"
     AND team."archivedAt" IS NULL
    WHERE org.id = org_id
      AND org."isPersonal"
      AND org."deletedAt" IS NULL
    ORDER BY team."createdAt", team.id
    LIMIT 1;
$$ LANGUAGE sql STABLE;

CREATE OR REPLACE FUNCTION normalize_personal_team()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW."organizationId" IS NOT NULL AND NEW."teamId" IS NULL THEN
        NEW."teamId" := personal_default_team_id(NEW."organizationId");
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- A key acts in its restriction, so a personal key's restriction and team are
-- both the slot it lives in.
CREATE OR REPLACE FUNCTION normalize_personal_api_key_team()
RETURNS TRIGGER AS $$
DECLARE
    default_team_id TEXT;
BEGIN
    IF NEW."organizationId" IS NULL
       OR (NEW."teamIdRestriction" IS NOT NULL
           AND NEW."teamId" IS NOT DISTINCT FROM NEW."teamIdRestriction") THEN
        RETURN NEW;
    END IF;
    default_team_id := personal_default_team_id(NEW."organizationId");
    IF default_team_id IS NULL THEN
        RETURN NEW;
    END IF;
    NEW."teamIdRestriction" := COALESCE(
        NEW."teamIdRestriction", NEW."teamId", default_team_id
    );
    NEW."teamId" := NEW."teamIdRestriction";
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $search_path$
DECLARE
    app_schema TEXT := current_schema();
BEGIN
    EXECUTE format(
        'ALTER FUNCTION %I.personal_default_team_id(TEXT) '
        'SET search_path = pg_catalog, %I, pg_temp',
        app_schema,
        app_schema
    );
    EXECUTE format(
        'ALTER FUNCTION %I.normalize_personal_team() '
        'SET search_path = pg_catalog, %I, pg_temp',
        app_schema,
        app_schema
    );
    EXECUTE format(
        'ALTER FUNCTION %I.normalize_personal_api_key_team() '
        'SET search_path = pg_catalog, %I, pg_temp',
        app_schema,
        app_schema
    );
END
$search_path$;

DO $triggers$
DECLARE
    tenant_table TEXT;
BEGIN
    FOREACH tenant_table IN ARRAY ARRAY[
        'AgentGraph',
        'AgentGraphExecution',
        'AgentPreset',
        'AlertCondition',
        'BuilderSearchHistory',
        'ChatSession',
        'Expert',
        'IntegrationWebhook',
        'LibraryAgent',
        'LibraryFolder',
        'PendingHumanReview',
        'StoreListingVersion',
        'UserNotificationBatch',
        'UserWorkspaceFile',
        'UserWorkspaceFolder'
    ] LOOP
        EXECUTE format(
            'DROP TRIGGER IF EXISTS a0_normalize_personal_team ON %I',
            tenant_table
        );
        EXECUTE format(
            'CREATE TRIGGER a0_normalize_personal_team '
            'BEFORE INSERT OR UPDATE OF "organizationId", "teamId" '
            'ON %I FOR EACH ROW EXECUTE FUNCTION normalize_personal_team()',
            tenant_table
        );
    END LOOP;
END
$triggers$;

DROP TRIGGER IF EXISTS a0_normalize_personal_team ON "APIKey";
CREATE TRIGGER a0_normalize_personal_team
BEFORE INSERT OR UPDATE OF "organizationId", "teamId", "teamIdRestriction"
ON "APIKey" FOR EACH ROW EXECUTE FUNCTION normalize_personal_api_key_team();

-- Who may hold rows in each slot, checked once here instead of per row: an
-- active member of both the personal organization and its default team. Rows
-- of anyone else stay where they are and are counted below.
CREATE TEMP TABLE personal_slot_member ON COMMIT DROP AS
SELECT DISTINCT ON (org.id, team_member."userId")
    org.id AS org_id,
    team.id AS team_id,
    team_member."userId" AS user_id
FROM "Organization" AS org
JOIN "Team" AS team
  ON team."orgId" = org.id
 AND team."isDefault"
 AND team."archivedAt" IS NULL
JOIN "TeamMember" AS team_member
  ON team_member."teamId" = team.id
 AND team_member.status = 'ACTIVE'
JOIN "OrgMember" AS org_member
  ON org_member."orgId" = org.id
 AND org_member."userId" = team_member."userId"
 AND org_member.status = 'ACTIVE'
WHERE org."isPersonal"
  AND org."deletedAt" IS NULL
ORDER BY org.id, team_member."userId", team."createdAt", team.id;
CREATE UNIQUE INDEX ON personal_slot_member (org_id, user_id);
ANALYZE personal_slot_member;

UPDATE "AgentGraph" AS resource
SET "teamId" = member.team_id
FROM personal_slot_member AS member
WHERE resource."organizationId" = member.org_id
  AND resource."userId" = member.user_id
  AND resource."teamId" IS NULL;

UPDATE "StoreListingVersion" AS version
SET "teamId" = graph."teamId"
FROM "AgentGraph" AS graph, personal_slot_member AS member
WHERE graph.id = version."agentGraphId"
  AND graph.version = version."agentGraphVersion"
  AND graph."organizationId" = member.org_id
  AND graph."userId" = member.user_id
  AND graph."teamId" = member.team_id
  AND version."organizationId" = member.org_id
  AND version."teamId" IS NULL;

-- A root folder whose name the default team already uses stays, with its
-- subfolders. Prod can't collide: names were unique per user, not per scope,
-- until 20261101130000.
CREATE TEMP TABLE personal_slot_kept_folder ON COMMIT DROP AS
WITH RECURSIVE kept AS (
    SELECT folder.id
    FROM "LibraryFolder" AS folder
    JOIN personal_slot_member AS member
      ON member.org_id = folder."organizationId"
     AND member.user_id = folder."userId"
    WHERE folder."teamId" IS NULL
      AND folder."parentId" IS NULL
      AND NOT folder."isDeleted"
      AND EXISTS (
          SELECT 1
          FROM "LibraryFolder" AS taken
          WHERE taken."userId" = folder."userId"
            AND taken."organizationId" = folder."organizationId"
            AND taken."teamId" = member.team_id
            AND taken."parentId" IS NULL
            AND NOT taken."isDeleted"
            AND taken.name = folder.name
      )
    UNION
    SELECT child.id
    FROM "LibraryFolder" AS child
    JOIN kept ON child."parentId" = kept.id
)
SELECT id FROM kept;

UPDATE "LibraryFolder" AS resource
SET "teamId" = member.team_id
FROM personal_slot_member AS member
WHERE resource."organizationId" = member.org_id
  AND resource."userId" = member.user_id
  AND resource."teamId" IS NULL
  AND resource.id NOT IN (SELECT id FROM personal_slot_kept_folder);

-- An entry stays when the default team already has the same version (the
-- entry key was unique per user until 20260826120000), when its folder stays,
-- or when it is the user's own graph and that version did not move with it.
UPDATE "LibraryAgent" AS resource
SET "teamId" = member.team_id
FROM personal_slot_member AS member
WHERE resource."organizationId" = member.org_id
  AND resource."userId" = member.user_id
  AND resource."teamId" IS NULL
  AND (
      resource."folderId" IS NULL
      OR resource."folderId" NOT IN (SELECT id FROM personal_slot_kept_folder)
  )
  AND NOT EXISTS (
      SELECT 1
      FROM "LibraryAgent" AS taken
      WHERE taken."userId" = resource."userId"
        AND taken."agentGraphId" = resource."agentGraphId"
        AND taken."agentGraphVersion" = resource."agentGraphVersion"
        AND taken."scopeKey" = member.org_id || ':' || member.team_id
  )
  AND (
      NOT resource."isCreatedByUser"
      OR EXISTS (
          SELECT 1
          FROM "AgentGraph" AS graph
          WHERE graph.id = resource."agentGraphId"
            AND graph.version = resource."agentGraphVersion"
            AND graph."userId" = resource."userId"
            AND graph."organizationId" = member.org_id
            AND graph."teamId" = member.team_id
      )
  );

UPDATE "AgentPreset" AS resource
SET "teamId" = member.team_id
FROM personal_slot_member AS member
WHERE resource."organizationId" = member.org_id
  AND resource."userId" = member.user_id
  AND resource."teamId" IS NULL;

UPDATE "IntegrationWebhook" AS resource
SET "teamId" = member.team_id
FROM personal_slot_member AS member
WHERE resource."organizationId" = member.org_id
  AND resource."userId" = member.user_id
  AND resource."teamId" IS NULL;

UPDATE "Expert" AS resource
SET "teamId" = member.team_id
FROM personal_slot_member AS member
WHERE resource."organizationId" = member.org_id
  AND resource."ownerUserId" = member.user_id
  AND resource."teamId" IS NULL;

-- The two large tables: membership was checked above, so the owner guard's
-- per-row share locks are skipped for the bulk move.
ALTER TABLE "ChatSession" DISABLE TRIGGER enforce_live_tenant_owner;
UPDATE "ChatSession" AS resource
SET "teamId" = member.team_id
FROM personal_slot_member AS member
WHERE resource."organizationId" = member.org_id
  AND resource."userId" = member.user_id
  AND resource."teamId" IS NULL;
ALTER TABLE "ChatSession" ENABLE TRIGGER enforce_live_tenant_owner;

ALTER TABLE "AgentGraphExecution" DISABLE TRIGGER enforce_live_tenant_owner;
UPDATE "AgentGraphExecution" AS resource
SET "teamId" = member.team_id
FROM personal_slot_member AS member
WHERE resource."organizationId" = member.org_id
  AND resource."userId" = member.user_id
  AND resource."teamId" IS NULL;
ALTER TABLE "AgentGraphExecution" ENABLE TRIGGER enforce_live_tenant_owner;

-- Workspace scope is immutable to ordinary writes, so the guards are held off
-- for this move and the scope key is set here, as 20261016173000 did. A root
-- folder whose name the default team already uses stays, with everything in it,
-- and so does a file whose path is taken there.
CREATE TEMP TABLE personal_slot_kept_workspace_folder ON COMMIT DROP AS
WITH RECURSIVE kept AS (
    SELECT folder.id
    FROM "UserWorkspaceFolder" AS folder
    JOIN "UserWorkspace" AS workspace ON workspace.id = folder."workspaceId"
    JOIN personal_slot_member AS member
      ON member.org_id = folder."organizationId"
     AND member.user_id = workspace."userId"
    WHERE folder."teamId" IS NULL
      AND folder."parentId" IS NULL
      AND NOT folder."isDeleted"
      AND EXISTS (
          SELECT 1
          FROM "UserWorkspaceFolder" AS taken
          WHERE taken."workspaceId" = folder."workspaceId"
            AND taken."scopeKey" = member.org_id || ':' || member.team_id
            AND taken."parentId" IS NULL
            AND NOT taken."isDeleted"
            AND taken.name = folder.name
      )
    UNION
    SELECT child.id
    FROM "UserWorkspaceFolder" AS child
    JOIN kept ON child."parentId" = kept.id
)
SELECT id FROM kept;

ALTER TABLE "UserWorkspaceFolder" DISABLE TRIGGER enforce_workspace_folder_scope;
UPDATE "UserWorkspaceFolder" AS folder
SET "teamId" = member.team_id,
    "scopeKey" = member.org_id || ':' || member.team_id
FROM "UserWorkspace" AS workspace, personal_slot_member AS member
WHERE workspace.id = folder."workspaceId"
  AND member.org_id = folder."organizationId"
  AND member.user_id = workspace."userId"
  AND folder."teamId" IS NULL
  AND folder."scopeResolved"
  AND folder.id NOT IN (SELECT id FROM personal_slot_kept_workspace_folder);
ALTER TABLE "UserWorkspaceFolder" ENABLE TRIGGER enforce_workspace_folder_scope;

ALTER TABLE "UserWorkspaceFile" DISABLE TRIGGER enforce_workspace_artifact_scope;
UPDATE "UserWorkspaceFile" AS file
SET "teamId" = member.team_id,
    "scopeKey" = member.org_id || ':' || member.team_id
FROM "UserWorkspace" AS workspace, personal_slot_member AS member
WHERE workspace.id = file."workspaceId"
  AND member.org_id = file."organizationId"
  AND member.user_id = workspace."userId"
  AND file."teamId" IS NULL
  AND file."scopeResolved"
  AND NOT file."isUserGlobalConfig"
  AND (
      file."folderId" IS NULL
      OR file."folderId" NOT IN (SELECT id FROM personal_slot_kept_workspace_folder)
  )
  AND (
      file."isDeleted"
      OR NOT EXISTS (
          SELECT 1
          FROM "UserWorkspaceFile" AS taken
          WHERE taken."workspaceId" = file."workspaceId"
            AND taken."scopeKey" = member.org_id || ':' || member.team_id
            AND taken.path = file.path
            AND NOT taken."isDeleted"
      )
  );
ALTER TABLE "UserWorkspaceFile" ENABLE TRIGGER enforce_workspace_artifact_scope;

-- Alert provenance is immutable to ordinary writes; alerts follow the
-- executions that moved above.
ALTER TABLE "AlertCondition" DISABLE TRIGGER "AlertCondition_execution_scope_guard";
UPDATE "AlertCondition" AS resource
SET "teamId" = member.team_id
FROM personal_slot_member AS member
WHERE resource."organizationId" = member.org_id
  AND resource."userId" = member.user_id
  AND resource."teamId" IS NULL;
ALTER TABLE "AlertCondition" ENABLE TRIGGER "AlertCondition_execution_scope_guard";

UPDATE "BuilderSearchHistory" AS resource
SET "teamId" = member.team_id
FROM personal_slot_member AS member
WHERE resource."organizationId" = member.org_id
  AND resource."userId" = member.user_id
  AND resource."teamId" IS NULL;

UPDATE "PendingHumanReview" AS resource
SET "teamId" = member.team_id
FROM personal_slot_member AS member
WHERE resource."organizationId" = member.org_id
  AND resource."userId" = member.user_id
  AND resource."teamId" IS NULL;

UPDATE "UserNotificationBatch" AS resource
SET "teamId" = member.team_id
FROM personal_slot_member AS member
WHERE resource."organizationId" = member.org_id
  AND resource."userId" = member.user_id
  AND resource."teamId" IS NULL;

-- A personal key acts in its restriction: fill it, and keep its team the same.
UPDATE "APIKey" AS key
SET "teamIdRestriction" = member.team_id,
    "teamId" = member.team_id
FROM personal_slot_member AS member
WHERE key."organizationId" = member.org_id
  AND key."userId" = member.user_id
  AND COALESCE(key."teamIdRestriction", key."teamId", member.team_id)
      = member.team_id
  AND (
      key."teamIdRestriction" IS NULL
      OR key."teamId" IS DISTINCT FROM key."teamIdRestriction"
  );

DO $left_behind$
DECLARE
    tenant_table TEXT;
    left_behind BIGINT;
BEGIN
    FOREACH tenant_table IN ARRAY ARRAY[
        'AgentGraph',
        'AgentGraphExecution',
        'AgentPreset',
        'AlertCondition',
        'APIKey',
        'BuilderSearchHistory',
        'ChatSession',
        'Expert',
        'IntegrationWebhook',
        'LibraryAgent',
        'LibraryFolder',
        'PendingHumanReview',
        'StoreListingVersion',
        'UserNotificationBatch',
        'UserWorkspaceFile',
        'UserWorkspaceFolder'
    ] LOOP
        EXECUTE format(
            'SELECT count(*) FROM %I AS resource '
            'JOIN "Organization" AS org ON org.id = resource."organizationId" '
            'WHERE org."isPersonal" AND org."deletedAt" IS NULL '
            'AND resource."teamId" IS NULL',
            tenant_table
        ) INTO left_behind;
        IF left_behind > 0 THEN
            RAISE NOTICE '% personal-organization rows in % stay at org-home',
                left_behind,
                tenant_table;
        END IF;
    END LOOP;
END
$left_behind$;

-- The team comparisons 20260827180000 left out, now that personal rows share
-- one slot. Pairs in different organizations are that audit's to refuse.
DO $strict$
DECLARE
    invalid_count BIGINT;
BEGIN
    SELECT count(*) INTO invalid_count
    FROM "APIKey" AS key
    WHERE key."teamId" IS DISTINCT FROM key."teamIdRestriction";
    IF invalid_count > 0 THEN
        RAISE EXCEPTION 'APIKey has % inconsistent workspace restrictions', invalid_count
            USING ERRCODE = '23514';
    END IF;

    SELECT count(*) INTO invalid_count
    FROM "AgentGraph" AS graph
    JOIN "AgentGraph" AS other
      ON other.id = graph.id AND other.version < graph.version
    WHERE other."organizationId" IS NOT DISTINCT FROM graph."organizationId"
      AND other."teamId" IS DISTINCT FROM graph."teamId";
    IF invalid_count > 0 THEN
        RAISE EXCEPTION 'AgentGraph has % version pairs with inconsistent tenancy',
            invalid_count
            USING ERRCODE = '23514';
    END IF;

    SELECT count(*) INTO invalid_count
    FROM "AgentGraphExecution" AS child
    JOIN "AgentGraphExecution" AS parent
      ON parent.id = child."parentGraphExecutionId"
    WHERE child."organizationId" IS NOT DISTINCT FROM parent."organizationId"
      AND child."teamId" IS DISTINCT FROM parent."teamId";
    IF invalid_count > 0 THEN
        RAISE EXCEPTION 'AgentGraphExecution has % parent-child scope mismatches',
            invalid_count
            USING ERRCODE = '23514';
    END IF;
END
$strict$;

COMMIT;
