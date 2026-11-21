-- Builder-saved GitHub Read File / Read Folder nodes captured the schema
-- default branch "main" into constantInput at render time. Once those blocks
-- send that value as ?ref=, a repository whose default branch is not main
-- 404s. Drop the key so the blocks omit ref and GitHub serves the repository
-- default branch. An explicit branch other than "main" is left alone.
--
-- A node whose repo_url is an api.github.com URL keeps its "main": the old
-- converter passed those URLs through with their query, so the node already
-- sent ref=main and read main. The pattern matches every form the converter
-- took for the API host (any case, padded, with userinfo or a port). A linked
-- repo_url can't be checked here, so that node is cleared like a github.com one.
--
-- Block IDs are GithubReadFileBlock and GithubReadFolderBlock in
-- backend/blocks/github/repo_files.py. constantInput is jsonb
-- (20250213110232_migrate_string_json).

UPDATE "AgentNode"
SET "constantInput" = "constantInput" - 'branch'
WHERE "agentBlockId" IN (
  '87ce6c27-5752-4bbc-8e26-6da40a3dcfd3', -- GithubReadFileBlock
  '1355f863-2db3-4d75-9fba-f91e8a8ca400'  -- GithubReadFolderBlock
)
AND "constantInput"->>'branch' = 'main'
AND COALESCE("constantInput"->>'repo_url', '')
  !~* '^\s*https?://([^/?#@]*@)?api\.github\.com(:[0-9]*)?([/?#]|$)';
