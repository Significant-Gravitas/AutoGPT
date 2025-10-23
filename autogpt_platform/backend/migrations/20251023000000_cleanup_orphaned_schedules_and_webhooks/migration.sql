-- CreateIndex: Add performance index for cleanup queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_library_agent_user_graph_active ON "LibraryAgent"("userId", "agentGraphId", "isDeleted", "isArchived");

-- Clean up orphaned APScheduler jobs for graphs not in users' libraries
-- WARNING: This migration assumes APScheduler stores job_state as JSON, but APScheduler
-- typically stores pickled binary data. If this migration fails with JSON parsing errors,
-- use the Python script approach in cleanup_orphaned_resources.py instead.
-- We need to extract the graph_id and user_id from the job_state->kwargs->graph_id and job_state->kwargs->user_id fields

-- First, create a temporary view to identify orphaned schedule jobs
-- This will fail gracefully if job_state is not JSON-parseable
DO $$
BEGIN
    -- Attempt to create the view for JSON-based job data
    EXECUTE '
    CREATE TEMPORARY VIEW orphaned_schedule_jobs AS
    SELECT 
        aj.id,
        (aj.job_state::json->''kwargs''->>''graph_id'') as graph_id,
        (aj.job_state::json->''kwargs''->>''user_id'') as user_id
    FROM apscheduler_jobs aj
    WHERE 
        -- Only process jobs that have the graph execution structure
        aj.job_state::json->''kwargs'' ? ''graph_id'' 
        AND aj.job_state::json->''kwargs'' ? ''user_id''
        -- Check if the graph is NOT in the user''s library (deleted/archived)
        AND NOT EXISTS (
            SELECT 1 FROM "LibraryAgent" la 
            WHERE la."userId" = (aj.job_state::json->''kwargs''->>''user_id'')
            AND la."agentGraphId" = (aj.job_state::json->''kwargs''->>''graph_id'')
            AND la."isDeleted" = false 
            AND la."isArchived" = false
        )';
EXCEPTION
    WHEN OTHERS THEN
        -- If JSON parsing fails, create an empty view and log a warning
        RAISE NOTICE 'APScheduler job_state is not JSON-parseable. Skipping schedule cleanup.';
        RAISE NOTICE 'Use the Python script cleanup_orphaned_resources.py for pickled data.';
        CREATE TEMPORARY VIEW orphaned_schedule_jobs AS
        SELECT '' as id, '' as graph_id, '' as user_id WHERE false;
END
$$;

-- Log the orphaned schedules we're about to clean up
DO $$
DECLARE
    orphaned_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO orphaned_count FROM orphaned_schedule_jobs;
    RAISE NOTICE 'Found % orphaned schedule jobs to clean up', orphaned_count;
END
$$;

-- Delete orphaned schedule jobs
DELETE FROM apscheduler_jobs 
WHERE id IN (SELECT id FROM orphaned_schedule_jobs);

-- Clean up orphaned webhook node triggers
-- Remove webhook triggers for nodes that belong to graphs not in users' libraries

-- First, log what we're about to clean up
DO $$
DECLARE
    orphaned_webhook_triggers INTEGER;
BEGIN
    SELECT COUNT(*) INTO orphaned_webhook_triggers
    FROM "WebhookToNode" wtn
    JOIN "AgentNode" an ON wtn."nodeId" = an."id"
    WHERE NOT EXISTS (
        SELECT 1 FROM "LibraryAgent" la 
        WHERE la."userId" = an."userId"
        AND la."agentGraphId" = an."agentGraphId"
        AND la."isDeleted" = false 
        AND la."isArchived" = false
    );
    RAISE NOTICE 'Found % orphaned webhook-to-node triggers to clean up', orphaned_webhook_triggers;
END
$$;

-- Delete orphaned webhook-to-node triggers
DELETE FROM "WebhookToNode" 
WHERE "nodeId" IN (
    SELECT an."id"
    FROM "AgentNode" an
    WHERE NOT EXISTS (
        SELECT 1 FROM "LibraryAgent" la 
        WHERE la."userId" = an."userId"
        AND la."agentGraphId" = an."agentGraphId"
        AND la."isDeleted" = false 
        AND la."isArchived" = false
    )
);

-- Clean up orphaned webhook preset triggers  
-- Remove webhook triggers for presets that belong to graphs not in users' libraries

-- First, log what we're about to clean up
DO $$
DECLARE
    orphaned_preset_webhooks INTEGER;
BEGIN
    SELECT COUNT(*) INTO orphaned_preset_webhooks
    FROM "AgentPreset" ap
    WHERE ap."webhookId" IS NOT NULL
    AND NOT EXISTS (
        SELECT 1 FROM "LibraryAgent" la 
        WHERE la."userId" = ap."userId"
        AND la."agentGraphId" = ap."agentGraphId"
        AND la."isDeleted" = false 
        AND la."isArchived" = false
    );
    RAISE NOTICE 'Found % orphaned preset webhook triggers to clean up', orphaned_preset_webhooks;
END
$$;

-- Remove webhook references from orphaned presets
UPDATE "AgentPreset" 
SET "webhookId" = NULL
WHERE "webhookId" IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM "LibraryAgent" la 
    WHERE la."userId" = "AgentPreset"."userId"
    AND la."agentGraphId" = "AgentPreset"."agentGraphId"
    AND la."isDeleted" = false 
    AND la."isArchived" = false
);

-- Clean up webhooks that no longer have any triggers
-- Delete webhooks that have no remaining triggered nodes or presets

-- First, log what we're about to clean up
DO $$
DECLARE
    dangling_webhooks INTEGER;
BEGIN
    SELECT COUNT(*) INTO dangling_webhooks
    FROM "Webhook" w
    WHERE NOT EXISTS (
        SELECT 1 FROM "WebhookToNode" wtn WHERE wtn."webhookId" = w."id"
    )
    AND NOT EXISTS (
        SELECT 1 FROM "AgentPreset" ap WHERE ap."webhookId" = w."id"
    );
    RAISE NOTICE 'Found % dangling webhooks with no triggers to clean up', dangling_webhooks;
END
$$;

-- Delete webhooks with no remaining triggers
DELETE FROM "Webhook" 
WHERE NOT EXISTS (
    SELECT 1 FROM "WebhookToNode" wtn WHERE wtn."webhookId" = "Webhook"."id"
)
AND NOT EXISTS (
    SELECT 1 FROM "AgentPreset" ap WHERE ap."webhookId" = "Webhook"."id"
);

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'Orphaned schedule and webhook cleanup completed successfully';
END
$$;