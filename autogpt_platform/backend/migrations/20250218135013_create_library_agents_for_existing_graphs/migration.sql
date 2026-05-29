-- Create LibraryAgents for all AgentGraphs in their owners' library, skipping existing entries
INSERT INTO "LibraryAgent" (
  "id",
  "createdAt",
  "updatedAt",
  "userId",
  "agentId",
  "agentVersion",
  "useGraphIsActiveVersion",
  "isFavorite",
  "isCreatedByUser",
  "isArchived",
  "isDeleted")
SELECT
  gen_random_uuid(), --> id
  ag."createdAt",    --> createdAt
  ag."createdAt",    --> updatedAt
  ag."userId",       --> userId
  ag."id",           --> agentId
  ag."version",      --> agentVersion
  true,              --> useGraphIsActiveVersion
  false,             --> isFavorite
  true,              --> isCreatedByUser
  false,             --> isArchived
  false              --> isDeleted
FROM  "AgentGraph" AS ag
WHERE ag."isActive" = true
AND   NOT EXISTS (
  SELECT 1
  FROM   "LibraryAgent" AS la
  WHERE  la."userId"  = ag."userId"
  AND    la."agentId" = ag."id"
);
