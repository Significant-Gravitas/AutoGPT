-- LibraryFolder had the same uniqueness gaps UserWorkspaceFolder already fixed:
-- Postgres treats NULLs as distinct, so the Prisma-generated
-- (userId, parentId, name) unique never constrained root folders (parentId IS
-- NULL), and soft-deleted rows kept their name reserved under a parent forever
-- because the index had no isDeleted predicate. Mirror the workspace-folder
-- partial indexes so a live name is unique among siblings and among roots,
-- and a soft-deleted name can be reused.

-- Rename any live root duplicates so the new root index can be created.
-- Nested-folder duplicates can't exist under the old composite unique.
-- The oldest row keeps its name; each later one takes the first free
-- "<name> (n)" among that user's live roots, the same shape
-- create_folder_with_unique_name produces. Probing the table before every
-- rename means a pre-existing "Work (2)" can never collide with a renamed
-- row, so CREATE UNIQUE INDEX below can't fail on the data this writes.
DO $$
DECLARE
  dup RECORD;
  candidate TEXT;
  n INT;
BEGIN
  FOR dup IN
    SELECT "id", "userId", "name", rn
    FROM (
      SELECT
        "id",
        "userId",
        "name",
        ROW_NUMBER() OVER (
          PARTITION BY "userId", "name"
          ORDER BY "createdAt", "id"
        ) AS rn
      FROM "LibraryFolder"
      WHERE "parentId" IS NULL AND "isDeleted" = false
    ) ranked
    WHERE rn > 1
    ORDER BY "userId", "name", rn
  LOOP
    n := dup.rn;
    LOOP
      candidate := dup."name" || ' (' || n || ')';
      EXIT WHEN NOT EXISTS (
        SELECT 1
        FROM "LibraryFolder"
        WHERE "userId" = dup."userId"
          AND "parentId" IS NULL
          AND "isDeleted" = false
          AND "name" = candidate
      );
      n := n + 1;
    END LOOP;
    UPDATE "LibraryFolder" SET "name" = candidate WHERE "id" = dup."id";
  END LOOP;
END $$;

DROP INDEX "LibraryFolder_userId_parentId_name_key";

CREATE UNIQUE INDEX "LibraryFolder_userId_parentId_name_live_key"
  ON "LibraryFolder"("userId", "parentId", "name")
  WHERE "isDeleted" = false;

CREATE UNIQUE INDEX "LibraryFolder_userId_name_root_live_key"
  ON "LibraryFolder"("userId", "name")
  WHERE "parentId" IS NULL AND "isDeleted" = false;
