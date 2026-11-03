-- 20261008180000_library_folder_name_unique_partial made a live root folder
-- name unique per user. Folders belong to one organization and team, and each
-- workspace lists only its own, so that index refused a root folder in one
-- workspace because the same user has one with the same name in another,
-- where it cannot be seen. Make the root rule per workspace. Nested names are
-- already per workspace: a folder's parent is in the folder's own workspace.

DROP INDEX "LibraryFolder_userId_name_root_live_key";

CREATE UNIQUE INDEX "LibraryFolder_userId_scope_name_root_live_key"
  ON "LibraryFolder"(
    "userId",
    COALESCE("organizationId", ''),
    COALESCE("teamId", ''),
    "name"
  )
  WHERE "parentId" IS NULL AND "isDeleted" = false;
