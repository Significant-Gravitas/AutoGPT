-- Otto keeps some files in the workspace only for itself: large tool outputs
-- it reads back, the build scratchpad its prompt asks for, the browser session
-- it restores. The user's file lists showed them beside what the user asked
-- for. The write path now marks them (backend/data/workspace_internal.py);
-- this column holds the mark, so lists can leave them out in SQL and still
-- page correctly.
ALTER TABLE "UserWorkspaceFile" ADD COLUMN "isInternal" BOOLEAN NOT NULL DEFAULT false;

-- Mark the rows written before the column, by the same rule the write path
-- applies: an explicit purpose decides; without one, a parked tool output
-- directly under a session's root and the two reserved working-file names
-- are internal. A user's upload never is.
UPDATE "UserWorkspaceFile"
SET "isInternal" = true
WHERE COALESCE("metadata"->>'origin', '') <> 'user-upload'
  AND (
    "metadata"->>'purpose' IN ('tool-output', 'working-file')
    OR (
      "metadata"->>'purpose' IS NULL
      AND (
        "path" ~* '^/sessions/[^/]+/tool-(outputs|results)/[^/]+$'
        OR LOWER("name") IN ('build_state.json', '._browser_state.json')
      )
    )
  );

-- Their search entries go too: writes no longer index internal files, and
-- global search reads the index directly, so an entry left from before would
-- still be found until the orphan cleanup came round.
DELETE FROM "UnifiedContentEmbedding" uce
USING "UserWorkspaceFile" uwf
WHERE uce."contentType" = 'WORKSPACE_FILE'
  AND uce."contentId" = uwf.id
  AND uwf."isInternal" = true;
