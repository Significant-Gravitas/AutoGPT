DO $$
DECLARE
    app_schema TEXT := current_schema();
    function_name TEXT;
BEGIN
    FOREACH function_name IN ARRAY ARRAY[
        'sync_library_agent_scope_key',
        'enforce_live_tenant_resource_owner',
        'lock_org_member_tenancy_change',
        'lock_team_member_tenancy_change',
        'enforce_store_listing_version_tenancy',
        'enforce_agent_graph_grant_tenancy',
        'enforce_owned_library_agent_tenancy',
        'lock_expert_workflow_graph',
        'enforce_live_team_member_owner',
        'enforce_workspace_artifact_scope',
        'enforce_workspace_folder_scope',
        'enforce_shared_workspace_file_scope',
        'enforce_alert_condition_scope'
    ]
    LOOP
        EXECUTE format(
            'ALTER FUNCTION %I.%I() SET search_path = pg_catalog, %I, pg_temp',
            app_schema,
            function_name,
            app_schema
        );
    END LOOP;
END $$;
