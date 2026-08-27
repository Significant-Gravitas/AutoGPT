ALTER TABLE "AlertCondition"
ADD COLUMN "organizationId" TEXT,
ADD COLUMN "teamId" TEXT,
ADD COLUMN "sourceGraphExecutionId" TEXT;

-- The pre-provenance rows can contain agent names, review counts and deep
-- links but cannot be tied back to a current authorization scope. Conditions
-- are re-derived by their emitters, so fail closed instead of treating them
-- as personal alerts after an offboarding.
DELETE FROM "AlertCondition";

-- Older morning briefings were assembled across every organization with no
-- persisted authorization provenance. New briefings are personal-scope only;
-- remove the unreadable legacy copies and their posted chat messages.
DELETE FROM "ChatMessage"
WHERE "metadata"->>'kind' = 'morning_briefing';

DELETE FROM "UserBriefing";

ALTER TABLE "StoreListingVersion"
ADD COLUMN "teamId" TEXT;

UPDATE "StoreListingVersion" AS version
SET "teamId" = graph."teamId"
FROM "AgentGraph" AS graph
WHERE graph."id" = version."agentGraphId"
  AND graph."version" = version."agentGraphVersion";

CREATE INDEX "StoreListingVersion_teamId_idx"
ON "StoreListingVersion"("teamId");

CREATE OR REPLACE FUNCTION enforce_store_listing_version_tenancy()
RETURNS TRIGGER AS $$
DECLARE
    graph_org_id TEXT;
    graph_team_id TEXT;
    listing_org_id TEXT;
BEGIN
    SELECT "organizationId", "teamId" INTO graph_org_id, graph_team_id
    FROM "AgentGraph"
    WHERE id = NEW."agentGraphId" AND version = NEW."agentGraphVersion"
    FOR UPDATE;
    SELECT "owningOrgId" INTO listing_org_id
    FROM "StoreListing"
    WHERE id = NEW."storeListingId";
    IF NEW."organizationId" IS DISTINCT FROM graph_org_id
       OR NEW."teamId" IS DISTINCT FROM graph_team_id
       OR NEW."organizationId" IS DISTINCT FROM listing_org_id THEN
        RAISE EXCEPTION 'listing version tenancy must match its graph and listing'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql
SET search_path = pg_catalog, platform, pg_temp;

DROP TRIGGER IF EXISTS enforce_listing_version_tenancy ON "StoreListingVersion";
CREATE TRIGGER enforce_listing_version_tenancy
BEFORE INSERT OR UPDATE OF "agentGraphId", "agentGraphVersion", "storeListingId", "organizationId", "teamId"
ON "StoreListingVersion" FOR EACH ROW EXECUTE FUNCTION enforce_store_listing_version_tenancy();

CREATE INDEX "AlertCondition_userId_organizationId_teamId_status_idx"
ON "AlertCondition"("userId", "organizationId", "teamId", "status");

CREATE INDEX "AlertCondition_sourceGraphExecutionId_idx"
ON "AlertCondition"("sourceGraphExecutionId");

ALTER TABLE "AlertCondition"
ADD CONSTRAINT "AlertCondition_sourceGraphExecutionId_fkey"
FOREIGN KEY ("sourceGraphExecutionId") REFERENCES "AgentGraphExecution"("id")
ON DELETE CASCADE ON UPDATE CASCADE;

CREATE OR REPLACE FUNCTION enforce_alert_condition_scope()
RETURNS TRIGGER AS $$
DECLARE
    execution_user_id TEXT;
    execution_organization_id TEXT;
    execution_team_id TEXT;
BEGIN
    IF TG_OP = 'UPDATE' AND (
        OLD."organizationId" IS DISTINCT FROM NEW."organizationId"
        OR OLD."teamId" IS DISTINCT FROM NEW."teamId"
        OR OLD."sourceGraphExecutionId" IS DISTINCT FROM NEW."sourceGraphExecutionId"
        OR OLD."userId" IS DISTINCT FROM NEW."userId"
    ) THEN
        RAISE EXCEPTION 'alert authorization provenance is immutable';
    END IF;

    IF NEW."teamId" IS NOT NULL AND NEW."organizationId" IS NULL THEN
        RAISE EXCEPTION 'alert team scope requires an organization';
    END IF;

    IF NEW."organizationId" IS NOT NULL
       AND NEW."sourceGraphExecutionId" IS NULL THEN
        RAISE EXCEPTION 'organization alert requires a source execution';
    END IF;

    IF TG_OP = 'UPDATE' OR NEW."sourceGraphExecutionId" IS NULL THEN
        RETURN NEW;
    END IF;

    SELECT "userId", "organizationId", "teamId"
    INTO execution_user_id, execution_organization_id, execution_team_id
    FROM "AgentGraphExecution"
    WHERE "id" = NEW."sourceGraphExecutionId"
      AND "isDeleted" = FALSE
    FOR SHARE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'alert source execution does not exist';
    END IF;

    IF NEW."userId" IS DISTINCT FROM execution_user_id
       OR NEW."organizationId" IS DISTINCT FROM execution_organization_id
       OR NEW."teamId" IS DISTINCT FROM execution_team_id THEN
        RAISE EXCEPTION 'alert source execution scope mismatch';
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql
SET search_path = pg_catalog, platform, pg_temp;

CREATE TRIGGER "AlertCondition_execution_scope_guard"
BEFORE INSERT OR UPDATE ON "AlertCondition"
FOR EACH ROW EXECUTE FUNCTION enforce_alert_condition_scope();
