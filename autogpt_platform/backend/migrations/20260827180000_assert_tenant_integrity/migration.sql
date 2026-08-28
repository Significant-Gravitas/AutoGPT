DO $audit$
DECLARE
    invalid_count BIGINT;
    owner_column TEXT;
    resource_table TEXT;
BEGIN
    FOR resource_table IN
        SELECT unnest(ARRAY[
            'UserWorkspaceFile',
            'UserWorkspaceFolder',
            'BuilderSearchHistory',
            'ChatSession',
            'AgentGraph',
            'AgentPreset',
            'AlertCondition',
            'UserNotificationBatch',
            'LibraryAgent',
            'LibraryFolder',
            'Expert',
            'AgentGraphExecution',
            'PendingHumanReview',
            'IntegrationWebhook',
            'StoreListingVersion',
            'APIKey',
            'AuditLog'
        ])
    LOOP
        EXECUTE format(
            'SELECT count(*) FROM %I AS resource '
            'LEFT JOIN "Team" AS team ON team.id = resource."teamId" '
            'WHERE resource."teamId" IS NOT NULL AND ('
            'resource."organizationId" IS NULL OR team.id IS NULL OR '
            'team."orgId" IS DISTINCT FROM resource."organizationId")',
            resource_table
        ) INTO invalid_count;
        IF invalid_count > 0 THEN
            RAISE EXCEPTION '% has % rows whose workspace does not belong to its organization',
                resource_table,
                invalid_count
                USING ERRCODE = '23514';
        END IF;
    END LOOP;

    FOR resource_table, owner_column IN
        SELECT * FROM (VALUES
            ('BuilderSearchHistory', 'userId'),
            ('ChatSession', 'userId'),
            ('AgentGraph', 'userId'),
            ('AgentPreset', 'userId'),
            ('AlertCondition', 'userId'),
            ('UserNotificationBatch', 'userId'),
            ('LibraryAgent', 'userId'),
            ('LibraryFolder', 'userId'),
            ('AgentGraphExecution', 'userId'),
            ('PendingHumanReview', 'userId'),
            ('IntegrationWebhook', 'userId'),
            ('APIKey', 'userId'),
            ('Expert', 'ownerUserId')
        ) AS resources(table_name, user_column)
    LOOP
        EXECUTE format(
            'SELECT count(*) FROM %I AS resource '
            'WHERE resource."organizationId" IS NOT NULL AND ('
            'resource.%I IS NULL OR NOT EXISTS ('
            'SELECT 1 FROM "OrgMember" AS member '
            'JOIN "Organization" AS org ON org.id = member."orgId" '
            'WHERE member."orgId" = resource."organizationId" '
            'AND member."userId" = resource.%I '
            'AND member.status = ''ACTIVE'' AND org."deletedAt" IS NULL) OR ('
            'resource."teamId" IS NOT NULL AND NOT EXISTS ('
            'SELECT 1 FROM "TeamMember" AS member '
            'JOIN "Team" AS team ON team.id = member."teamId" '
            'WHERE member."teamId" = resource."teamId" '
            'AND member."userId" = resource.%I '
            'AND member.status = ''ACTIVE'' '
            'AND team."orgId" = resource."organizationId" '
            'AND team."archivedAt" IS NULL)))',
            resource_table,
            owner_column,
            owner_column,
            owner_column
        ) INTO invalid_count;
        IF invalid_count > 0 THEN
            RAISE EXCEPTION '% has % tenant rows without a live owner membership',
                resource_table,
                invalid_count
                USING ERRCODE = '23514';
        END IF;
    END LOOP;

    SELECT count(*) INTO invalid_count
    FROM "APIKey" AS key
    LEFT JOIN "Team" AS restricted_team
      ON restricted_team.id = key."teamIdRestriction"
    WHERE key."teamId" IS DISTINCT FROM key."teamIdRestriction"
       OR (
           key."teamIdRestriction" IS NOT NULL
           AND (
               key."organizationId" IS NULL
               OR restricted_team.id IS NULL
               OR restricted_team."orgId" IS DISTINCT FROM key."organizationId"
           )
       );
    IF invalid_count > 0 THEN
        RAISE EXCEPTION 'APIKey has % inconsistent workspace restrictions', invalid_count
            USING ERRCODE = '23514';
    END IF;

    SELECT count(*) INTO invalid_count
    FROM "OAuthApplication" AS application
    LEFT JOIN "Team" AS restricted_team
      ON restricted_team.id = application."teamIdRestriction"
    WHERE application."teamIdRestriction" IS NOT NULL
      AND (
          application."organizationId" IS NULL
          OR restricted_team.id IS NULL
          OR restricted_team."orgId" IS DISTINCT FROM application."organizationId"
      );
    IF invalid_count > 0 THEN
        RAISE EXCEPTION 'OAuthApplication has % inconsistent workspace restrictions',
            invalid_count
            USING ERRCODE = '23514';
    END IF;

    SELECT count(*) INTO invalid_count
    FROM "AgentGraph" AS graph
    JOIN "AgentGraph" AS other
      ON other.id = graph.id AND other.version < graph.version
    WHERE other."userId" IS DISTINCT FROM graph."userId"
       OR other."organizationId" IS DISTINCT FROM graph."organizationId"
       OR other."teamId" IS DISTINCT FROM graph."teamId";
    IF invalid_count > 0 THEN
        RAISE EXCEPTION 'AgentGraph has % version pairs with inconsistent tenancy',
            invalid_count
            USING ERRCODE = '23514';
    END IF;

    SELECT count(*) INTO invalid_count
    FROM "AgentGraphExecution" AS child
    JOIN "AgentGraphExecution" AS parent
      ON parent.id = child."parentGraphExecutionId"
    WHERE child."userId" IS DISTINCT FROM parent."userId"
       OR child."organizationId" IS DISTINCT FROM parent."organizationId"
       OR child."teamId" IS DISTINCT FROM parent."teamId";
    IF invalid_count > 0 THEN
        RAISE EXCEPTION 'AgentGraphExecution has % parent-child scope mismatches',
            invalid_count
            USING ERRCODE = '23514';
    END IF;

    SELECT count(*) INTO invalid_count
    FROM "StoreListingVersion" AS version
    JOIN "AgentGraph" AS graph
      ON graph.id = version."agentGraphId"
     AND graph.version = version."agentGraphVersion"
    JOIN "StoreListing" AS listing ON listing.id = version."storeListingId"
    WHERE version."organizationId" IS DISTINCT FROM graph."organizationId"
       OR version."teamId" IS DISTINCT FROM graph."teamId"
       OR listing."agentGraphId" IS DISTINCT FROM graph.id
       OR listing."owningUserId" IS DISTINCT FROM graph."userId"
       OR listing."owningOrgId" IS DISTINCT FROM graph."organizationId";
    IF invalid_count > 0 THEN
        RAISE EXCEPTION 'StoreListingVersion has % graph or listing scope mismatches',
            invalid_count
            USING ERRCODE = '23514';
    END IF;

    SELECT count(*) INTO invalid_count
    FROM "StoreListing" AS listing
    JOIN "StoreListingVersion" AS active_version
      ON active_version.id = listing."activeVersionId"
    WHERE active_version."storeListingId" IS DISTINCT FROM listing.id;
    IF invalid_count > 0 THEN
        RAISE EXCEPTION 'StoreListing has % active versions owned by another listing',
            invalid_count
            USING ERRCODE = '23514';
    END IF;
END
$audit$;

ALTER TABLE "ChatSession"
  DROP CONSTRAINT "ChatSession_teamId_fkey",
  ADD CONSTRAINT "ChatSession_teamId_fkey"
    FOREIGN KEY ("teamId") REFERENCES "Team"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
ALTER TABLE "AgentGraph"
  DROP CONSTRAINT "AgentGraph_teamId_fkey",
  ADD CONSTRAINT "AgentGraph_teamId_fkey"
    FOREIGN KEY ("teamId") REFERENCES "Team"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
ALTER TABLE "AgentPreset"
  DROP CONSTRAINT "AgentPreset_teamId_fkey",
  ADD CONSTRAINT "AgentPreset_teamId_fkey"
    FOREIGN KEY ("teamId") REFERENCES "Team"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
ALTER TABLE "LibraryAgent"
  DROP CONSTRAINT "LibraryAgent_teamId_fkey",
  ADD CONSTRAINT "LibraryAgent_teamId_fkey"
    FOREIGN KEY ("teamId") REFERENCES "Team"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
ALTER TABLE "LibraryFolder"
  DROP CONSTRAINT "LibraryFolder_teamId_fkey",
  ADD CONSTRAINT "LibraryFolder_teamId_fkey"
    FOREIGN KEY ("teamId") REFERENCES "Team"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
ALTER TABLE "Expert"
  DROP CONSTRAINT "Expert_teamId_fkey",
  ADD CONSTRAINT "Expert_teamId_fkey"
    FOREIGN KEY ("teamId") REFERENCES "Team"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
ALTER TABLE "AgentGraphExecution"
  DROP CONSTRAINT "AgentGraphExecution_teamId_fkey",
  ADD CONSTRAINT "AgentGraphExecution_teamId_fkey"
    FOREIGN KEY ("teamId") REFERENCES "Team"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
ALTER TABLE "IntegrationWebhook"
  DROP CONSTRAINT "IntegrationWebhook_teamId_fkey",
  ADD CONSTRAINT "IntegrationWebhook_teamId_fkey"
    FOREIGN KEY ("teamId") REFERENCES "Team"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
