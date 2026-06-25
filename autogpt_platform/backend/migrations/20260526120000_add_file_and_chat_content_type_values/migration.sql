-- Step 1/2: add WORKSPACE_FILE and CHAT_SESSION to the ContentType enum.
--
-- Postgres requires `ALTER TYPE ... ADD VALUE` to be committed before the
-- new value can be referenced in queries or constraints. Splitting this
-- into its own migration keeps the follow-up migration
-- (`20260526120001_user_scoped_content_type_check`) free to use the new
-- values in `DELETE` / `CHECK` expressions.

ALTER TYPE "ContentType" ADD VALUE IF NOT EXISTS 'WORKSPACE_FILE';
ALTER TYPE "ContentType" ADD VALUE IF NOT EXISTS 'CHAT_SESSION';
