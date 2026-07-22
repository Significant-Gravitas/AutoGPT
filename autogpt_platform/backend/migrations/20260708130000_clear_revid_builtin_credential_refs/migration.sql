-- The built-in "Use Credits for Revid" credential
-- (id fdb7f412-f519-48d1-9b5f-d2f73d0e01fe) has been removed. Revid blocks now
-- require a user-provided API key. Clear every stale reference to it so
-- affected graphs/presets show an empty "select a credential" state instead of
-- a dead built-in reference.

-- 1) Saved graph definitions: blank the node's credentials selection. `{}` is
-- how the app itself represents a cleared node credential (input_default = {}).
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{credentials}',
         '{}'::jsonb
       )
WHERE  "constantInput"::jsonb->'credentials'->>'id'
       = 'fdb7f412-f519-48d1-9b5f-d2f73d0e01fe';

-- 2) Preset input overrides only (agentPresetId IS NOT NULL — never execution
-- history). Each preset credential is its own AgentNodeExecutionInputOutput row
-- whose `data` IS the credential meta, read back via
-- CredentialsMetaInput.model_validate(). Blanking to `{}` would fail that
-- validation and break preset loading, so DELETE the stale row instead — the
-- preset then prompts for a user-provided key.
DELETE FROM "AgentNodeExecutionInputOutput"
WHERE  "agentPresetId" IS NOT NULL
  AND  "data"::jsonb->>'id' = 'fdb7f412-f519-48d1-9b5f-d2f73d0e01fe'
  AND  "data"::jsonb->>'provider' = 'revid';
