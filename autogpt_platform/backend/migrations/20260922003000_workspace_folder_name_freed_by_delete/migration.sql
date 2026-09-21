-- A soft-deleted nested folder kept its name reserved under its parent forever:
-- the Prisma-generated composite unique index carried no isDeleted predicate,
-- while the root index added with the table has always excluded deleted rows.
-- Nesting makes that asymmetry reachable — deleting "Invoices/2026" and
-- creating it again answered 409 — so replace it with a partial index shaped
-- like the root one. Named differently, and dropped from schema.prisma, so
-- Prisma stops expecting an unqualified index it cannot express.
DROP INDEX "UserWorkspaceFolder_workspaceId_parentId_name_key";

CREATE UNIQUE INDEX "UserWorkspaceFolder_workspaceId_parentId_name_live_key" ON "UserWorkspaceFolder"("workspaceId", "parentId", "name") WHERE "isDeleted" = false;
