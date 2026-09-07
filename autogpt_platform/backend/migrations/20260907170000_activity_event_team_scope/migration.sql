ALTER TABLE "ActivityEvent" ADD COLUMN "teamId" TEXT;

WITH scoped_events AS (
    SELECT event.id,
           COALESCE(execution."organizationId", session."organizationId") AS org_id,
           COALESCE(execution."teamId", session."teamId") AS team_id
    FROM "ActivityEvent" event
    LEFT JOIN "ChatSession" session
      ON session.id = event."sessionId" AND session."userId" = event."userId"
    LEFT JOIN "AgentGraphExecution" execution
      ON execution.id = event."graphExecId" AND execution."userId" = event."userId"
    JOIN "Organization" o
      ON o.id = COALESCE(execution."organizationId", session."organizationId")
     AND o."deletedAt" IS NULL
    WHERE COALESCE(execution."organizationId", session."organizationId") IS NOT NULL
      AND (event."sessionId" IS NULL OR session.id IS NOT NULL)
      AND (event."graphExecId" IS NULL OR execution.id IS NOT NULL)
      AND (
          session.id IS NULL OR execution.id IS NULL OR (
              session."organizationId" IS NOT DISTINCT FROM execution."organizationId"
              AND session."teamId" IS NOT DISTINCT FROM execution."teamId"
          )
      )
)
UPDATE "ActivityEvent" event
SET "organizationId" = source.org_id, "teamId" = source.team_id
FROM scoped_events source
WHERE event.id = source.id
  AND (event."organizationId" IS NULL OR event."organizationId" = source.org_id);

CREATE INDEX "ActivityEvent_userId_organizationId_teamId_createdAt_idx"
ON "ActivityEvent"("userId", "organizationId", "teamId", "createdAt");
