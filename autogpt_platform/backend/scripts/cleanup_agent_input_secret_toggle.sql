-- One-off cleanup: drop the obsolete `secret` and `value` keys from
-- AgentInput* nodes that had the (now removed) secret toggle enabled.
--
-- Run this once, after the release that removes the toggle has shipped.
-- Affected users were emailed a week in advance with a heads-up to
-- migrate any sensitive defaults out of node configs.
--
-- The AgentInput block IDs below cover the base block plus every
-- subclass declared in `backend/blocks/io.py`.

UPDATE "AgentNode"
SET "constantInput" = "constantInput" - 'secret' - 'value'
WHERE "agentBlockId" IN (
  'c0a8e994-ebf1-4a9c-a4d8-89d09c86741b', -- AgentInputBlock (base)
  '7fcd3bcb-8e1b-4e69-903d-32d3d4a92158', -- AgentShortTextInputBlock
  '90a56ffb-7024-4b2b-ab50-e26c5e5ab8ba', -- AgentLongTextInputBlock
  '96dae2bb-97a2-41c2-bd2f-13a3b5a8ea98', -- AgentNumberInputBlock
  '7e198b09-4994-47db-8b4d-952d98241817', -- AgentDateInputBlock
  '2a1c757e-86cf-4c7e-aacf-060dc382e434', -- AgentTimeInputBlock
  '95ead23f-8283-4654-aef3-10c053b74a31', -- AgentFileInputBlock
  '655d6fdf-a334-421c-b733-520549c07cd1', -- AgentDropdownInputBlock
  'cbf36ab5-df4a-43b6-8a7f-f7ed8652116e', -- AgentToggleInputBlock
  '5603b273-f41e-4020-af7d-fbc9c6a8d928', -- AgentTableInputBlock
  'd3b32f15-6fd7-40e3-be52-e083f51b19a2'  -- AgentGoogleDriveFileInputBlock
)
AND ("constantInput"->>'secret')::boolean IS TRUE;
