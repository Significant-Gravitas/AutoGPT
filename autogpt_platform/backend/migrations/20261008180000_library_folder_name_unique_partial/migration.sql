-- LibraryFolder had the same uniqueness gaps UserWorkspaceFolder already fixed:
-- Postgres treats NULLs as distinct, so the Prisma-generated
-- (userId, parentId, name) unique never constrained root folders (parentId IS
-- NULL), and soft-deleted rows kept their name reserved under a parent forever
-- because the index had no isDeleted predicate. Mirror the workspace-folder
-- partial indexes so a live name is unique among siblings and among roots,
-- and a soft-deleted name can be reused.

-- Rename any live root duplicates so the new root index can be created.
-- Nested-folder duplicates can't exist under the old composite unique.
WITH ranked AS (
  SELECT
    "id",
    "name",
    ROW_NUMBER() OVER (
      PARTITION BY "userId", "name"
      ORDER BY "createdAt", "id"
    ) AS rn
  FROM "LibraryFolder"
  WHERE "parentId" IS NULL AND "isDeleted" = false
)
UPDATE "LibraryFolder" AS f
SET "name" = f."name" || ' (' || ranked.rn || ')'
FROM ranked
WHERE f."id" = ranked."id" AND ranked.rn > 1;

DROP INDEX "LibraryFolder_userId_parentId_name_key";

CREATE UNIQUE INDEX "LibraryFolder_userId_parentId_name_live_key"
  ON "LibraryFolder"("userId", "parentId", "name")
  WHERE "isDeleted" = false;

CREATE UNIQUE INDEX "LibraryFolder_userId_name_root_live_key"
  ON "LibraryFolder"("userId", "name")
  WHERE "parentId" IS NULL AND "isDeleted" = false;
