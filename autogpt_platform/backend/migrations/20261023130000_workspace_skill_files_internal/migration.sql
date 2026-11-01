-- Hiring an expert installs its skill packages (SKILL.md, references, LICENSE,
-- ATTRIBUTION.md) in the user's workspace, and saving a skill writes one there
-- too. Files listed them beside what the user asked for. The write path now
-- marks every file in a skill folder internal (backend/data/workspace_internal.py);
-- this marks the ones written before it, by the same rule: a path under the
-- account's /skills/ or an expert's /experts/<id>/skills/, unless the user
-- uploaded it.
UPDATE "UserWorkspaceFile"
SET "isInternal" = true
WHERE NOT "isInternal"
  AND COALESCE("metadata"->>'origin', '') <> 'user-upload'
  AND ("path" LIKE '/skills/%' OR "path" ~ '^/experts/[^/]+/skills/');

-- Their search entries go too, as for Otto's working files: writes no longer
-- index skill files, and global search reads the index directly. This covers
-- every internal file, not only skills; it is idempotent with #16028's.
DELETE FROM "UnifiedContentEmbedding" uce
USING "UserWorkspaceFile" uwf
WHERE uce."contentType" = 'WORKSPACE_FILE'
  AND uce."contentId" = uwf.id
  AND uwf."isInternal" = true;
