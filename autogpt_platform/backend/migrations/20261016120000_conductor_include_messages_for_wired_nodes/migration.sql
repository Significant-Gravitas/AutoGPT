-- The Conductor wait/poll blocks stopped emitting their raw `messages` rows by
-- default; a new `include_messages` input (default false) opts back in. Saved
-- graphs that already wire `messages` downstream predate that input, so their
-- nodes would resolve it to false and the downstream node would silently stop
-- running. Turn it on for exactly those nodes so they keep their behaviour.
--
-- A link's sourceName is either the pin itself or a dynamic sub-pin
-- (`messages_$_0`, `messages_#_key`, `messages_@_attr`). Nodes that already
-- carry an explicit `include_messages` are left alone.

UPDATE "AgentNode" AS n
SET    "constantInput" = JSONB_SET(
         n."constantInput"::jsonb,
         '{include_messages}',
         'true'::jsonb,
         true
       )
WHERE  n."agentBlockId" IN (
         '4f22dbe5-2855-498a-ab82-6857441ea4b1', -- ConductorSendMessageBlock
         '3d4b2ccb-d654-4b5d-9750-e67f393f53f0', -- ConductorCreateSessionBlock
         'd419c562-873c-4a67-b83f-9bcf11254f3f', -- ConductorCreateWorkspaceBlock
         'fabe3db0-fb32-4b05-8e68-b169477b02ec'  -- ConductorGetSessionBlock
       )
  AND  NOT (n."constantInput"::jsonb ? 'include_messages')
  AND  EXISTS (
         SELECT 1
         FROM   "AgentNodeLink" AS l
         WHERE  l."agentNodeSourceId" = n."id"
           AND  l."sourceName" ~ '^messages(_[$#@]_.*)?$'
       );
