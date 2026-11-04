-- A soft-deleted nested folder kept its name reserved under its parent forever:
-- the Prisma-generated composite unique index carried no isDeleted predicate,
-- while the root index added with the table has always excluded deleted rows.
-- Nesting makes that asymmetry reachable — deleting "Invoices/2026" and
-- creating it again answered 409 — so replace it with a partial index shaped
-- like the root one. Named differently, and dropped from schema.prisma, so
-- Prisma stops expecting an unqualified index it cannot express.
DROP INDEX "UserWorkspaceFolder_workspaceId_parentId_name_key";

CREATE UNIQUE INDEX "UserWorkspaceFolder_workspaceId_parentId_name_live_key" ON "UserWorkspaceFolder"("workspaceId", "parentId", "name") WHERE "isDeleted" = false;

-- Files reserved their path the same way, and bought the freedom back in code:
-- soft-delete renamed the row ("/a.txt" -> "/a.txt__deleted__<ts>"). That put
-- the guarantee in one function rather than the schema, and it had already
-- failed once — a second-resolution suffix gave two deletes in one second the
-- same name, so the second rename hit the very index it exists to dodge. Give
-- files the folders' constraint instead, and let a deleted row keep the path
-- it was deleted at.
DROP INDEX "UserWorkspaceFile_workspaceId_path_key";

CREATE UNIQUE INDEX "UserWorkspaceFile_workspaceId_path_live_key" ON "UserWorkspaceFile"("workspaceId", "path") WHERE "isDeleted" = false;

-- Give the rows the old rename mangled their paths back: the index skips
-- deleted rows, so a restored path collides with nothing. Both suffixes ever
-- written are matched, the integer timestamp and the microsecond one.
UPDATE "UserWorkspaceFile"
SET "path" = regexp_replace("path", '__deleted__[0-9]+(\.[0-9]+)?$', '')
WHERE "isDeleted" = true AND "path" ~ '__deleted__[0-9]+(\.[0-9]+)?$';
