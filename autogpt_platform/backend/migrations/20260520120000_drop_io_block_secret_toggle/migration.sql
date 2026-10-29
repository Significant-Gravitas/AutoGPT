-- For AgentInput* / AgentOutput nodes where `constantInput.secret = true`,
-- drop both the `secret` key (no longer part of the block schema) and
-- the `value` key (the default value the user explicitly marked as
-- secret, which would otherwise become visible without the masking
-- toggle).
--
-- Block IDs below cover the AgentInput base block, every subclass
-- declared in `backend/blocks/io.py`, and the AgentOutput block.

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
  'd3b32f15-6fd7-40e3-be52-e083f51b19a2', -- AgentGoogleDriveFileInputBlock
  '363ae599-353e-4804-937e-b2ee3cef3da4'  -- AgentOutputBlock
)
AND "constantInput"->'secret' = 'true'::jsonb;
