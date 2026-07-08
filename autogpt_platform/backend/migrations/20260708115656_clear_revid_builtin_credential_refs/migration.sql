-- The built-in "Use Credits for Revid" credential
-- (id fdb7f412-f519-48d1-9b5f-d2f73d0e01fe) has been removed. Revid blocks
-- now require a user-provided API key. Clear the stale reference from every
-- node that still points at it so the builder shows an empty
-- "select a credential" state instead of a dead built-in reference. This
-- mirrors how the app itself clears stale credentials (input_default = {}).
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{credentials}',
         '{}'::jsonb
       )
WHERE  "constantInput"::jsonb->'credentials'->>'id'
       = 'fdb7f412-f519-48d1-9b5f-d2f73d0e01fe';
