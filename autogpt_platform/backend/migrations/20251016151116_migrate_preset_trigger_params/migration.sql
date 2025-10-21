/*
  Migration to support input blocks alongside webhook trigger blocks.

  This migration converts existing triggered presets to use the new format where:
  - Regular graph input values are stored as-is in AgentNodeExecutionInputOutput
  - Trigger-specific parameters are stored under a special key: _trigger_params_{node_prefix}

  This allows graphs to have both trigger blocks and input blocks simultaneously.
*/

BEGIN;
SET LOCAL statement_timeout = '10min';

-- Find all graphs with webhook trigger nodes (triggered graphs)
-- NOTE: Must check graph structure, not just preset.webhookId, because
-- presets can be auto-disabled (webhookId = NULL) while still being triggered presets
WITH triggered_graphs AS (
    SELECT DISTINCT
        an."agentGraphId" as graph_id,
        an."agentGraphVersion" as graph_version,
        an."id" as webhook_node_id,
        SPLIT_PART(an."id", '-', 1) as node_prefix
    FROM "AgentNode" an
    WHERE an."agentBlockId" IN (
        'd0180ce6-ccb9-48c7-8256-b39e93e62801', -- Airtable Webhook Trigger block
        '9464a020-ed1d-49e1-990f-7f2ac924a2b7', -- Compass AI Trigger block
        'd0204ed8-8b81-408d-8b8d-ed087a546228', -- Exa Webset Webhook block
        '8fa8c167-2002-47ce-aba8-97572fc5d387', -- Generic Webhook Trigger block
        '6c60ec01-8128-419e-988f-96a063ee2fea', -- GitHub Pull Request Trigger block
        '8a74c2ad-0104-4640-962f-26c6b69e58cd'  -- Slant3D Order Webhook block
    )
),

-- Find all presets using triggered graphs (both active and auto-disabled)
triggered_presets AS (
    SELECT
        ap."id" as preset_id,
        tg.graph_id,
        tg.graph_version,
        tg.webhook_node_id,
        tg.node_prefix,
        ap."webhookId" -- May be NULL if auto-disabled
    FROM "AgentPreset" ap
    JOIN triggered_graphs tg ON tg.graph_id = ap."agentGraphId"
                            AND tg.graph_version = ap."agentGraphVersion"
    WHERE ap."isDeleted" = false
),

-- Get all current input data for triggered presets
current_inputs AS (
    SELECT
        tp.preset_id,
        tp.node_prefix,
        aneio."name" as input_name,
        aneio."data" as input_data
    FROM triggered_presets tp
    JOIN "AgentNodeExecutionInputOutput" aneio ON aneio."id" = ANY(
        SELECT unnest(ap."InputPresets")
        FROM "AgentPreset" ap
        WHERE ap."id" = tp.preset_id
    )
)

-- Create new trigger parameter entries and link them to presets
INSERT INTO "AgentNodeExecutionInputOutput" ("id", "name", "data", "agentPresetId")
SELECT
    gen_random_uuid()::text,
    '_trigger_params_' || ci.node_prefix,
    jsonb_object_agg(ci.input_name, ci.input_data),
    ci.preset_id
FROM current_inputs ci
GROUP BY ci.preset_id, ci.node_prefix;

-- Note: This migration converts ALL existing inputs in triggered presets to trigger parameters
-- In the new system, regular graph inputs would be stored alongside these trigger parameters
-- but without the special _trigger_params_ prefix

COMMIT;