import re
from pathlib import Path

import pytest

from backend.data.db import prisma
from backend.data.org_migration import ensure_personal_org

_SEARCH_PATH_FUNCTIONS = {
    "sync_library_agent_scope_key",
    "enforce_live_tenant_resource_owner",
    "lock_org_member_tenancy_change",
    "lock_team_member_tenancy_change",
    "enforce_store_listing_version_tenancy",
    "enforce_agent_graph_grant_tenancy",
    "enforce_owned_library_agent_tenancy",
    "lock_expert_workflow_graph",
    "enforce_live_team_member_owner",
    "enforce_workspace_artifact_scope",
    "enforce_workspace_folder_scope",
    "enforce_shared_workspace_file_scope",
    "enforce_alert_condition_scope",
}


def _migration_sql() -> str:
    return (
        Path(__file__).parents[4]
        / "migrations"
        / "20260826120000_org_single_owner"
        / "migration.sql"
    ).read_text(encoding="utf-8")


def _workspace_migration_sql() -> str:
    return (
        Path(__file__).parents[4]
        / "migrations"
        / "20260826180000_scope_workspace_artifacts"
        / "migration.sql"
    ).read_text(encoding="utf-8")


def _notification_migration_sql() -> str:
    return (
        Path(__file__).parents[4]
        / "migrations"
        / "20260826210000_scope_execution_notifications"
        / "migration.sql"
    ).read_text(encoding="utf-8")


def _validation_lock_migration_sql() -> str:
    return (
        Path(__file__).parents[4]
        / "migrations"
        / "20260827120000_share_tenancy_validation_locks"
        / "migration.sql"
    ).read_text(encoding="utf-8")


def _search_path_repair_migration_sql() -> str:
    return (
        Path(__file__).parents[4]
        / "migrations"
        / "20260827130000_schema_safe_trigger_search_paths"
        / "migration.sql"
    ).read_text(encoding="utf-8")


def test_migration_enforces_single_live_org_owner_and_invitation() -> None:
    sql = _migration_sql()

    assert 'PARTITION BY "orgId"' in sql
    assert '"OrgMember_one_owner_per_org"' in sql
    assert 'WHERE "isOwner" = true' in sql
    assert 'PARTITION BY "orgId", LOWER(email)' in sql
    assert '"OrgInvitation_one_pending_per_email"' in sql


def test_migration_enforces_one_open_transfer_per_resource() -> None:
    sql = _migration_sql()

    assert 'PARTITION BY "resourceType", "resourceId"' in sql
    assert '"TransferRequest_one_open_per_resource"' in sql
    assert "status IN ('PENDING', 'SOURCE_APPROVED', 'TARGET_APPROVED')" in sql
    assert "SET status = 'REJECTED'" in sql


def test_migration_keeps_library_scope_key_in_sync() -> None:
    sql = _migration_sql()

    assert '"LibraryAgent_userId_agentGraphId_agentGraphVersion_scopeKey_key"' in sql
    assert "CREATE OR REPLACE FUNCTION sync_library_agent_scope_key()" in sql
    assert 'BEFORE INSERT OR UPDATE OF "organizationId", "teamId"' in sql


def test_migration_installs_live_owner_guards_on_durable_resources() -> None:
    sql = _migration_sql()

    for table in (
        "AgentGraph",
        "AgentGraphExecution",
        "LibraryAgent",
        "LibraryFolder",
        "AgentPreset",
        "IntegrationWebhook",
        "ChatSession",
    ):
        assert f"'{table}'" in sql
    assert 'ON "Expert" FOR EACH ROW' in sql
    assert 'ON "APIKey" FOR EACH ROW' in sql
    assert 'ON "TeamMember" FOR EACH ROW' in sql
    assert 'ON "StoreListing" FOR EACH ROW' in sql
    assert 'ON "StoreListingVersion" FOR EACH ROW' in sql
    assert "store listing tenancy must match its graph" in sql
    assert "listing version tenancy must match its graph and listing" in sql
    assert 'UPDATE OF "userId", "organizationId", "teamId", "executionStatus"' in sql


def test_migration_serializes_membership_changes_before_resource_writes() -> None:
    sql = _migration_sql()

    assert "CREATE OR REPLACE FUNCTION lock_org_member_tenancy_change()" in sql
    assert "CREATE OR REPLACE FUNCTION lock_team_member_tenancy_change()" in sql
    assert "tenancy:org-user:" in sql
    assert "tenancy:team:" in sql
    assert 'PERFORM 1 FROM "User" WHERE id = member_user_id FOR UPDATE' in sql
    assert 'PERFORM 1 FROM "Organization" WHERE id = member_org_id FOR UPDATE' in sql
    assert 'PERFORM 1 FROM "Team" WHERE id = member_team_id FOR UPDATE' in sql
    assert sql.count("CREATE TRIGGER a_lock_live_tenancy_change") == 2


def test_migration_trigger_search_paths_are_repaired_for_the_active_schema() -> None:
    sql = _search_path_repair_migration_sql()
    function_names = set(re.findall(r"^\s*'([a-z_]+)',?\s*$", sql, flags=re.MULTILINE))

    assert "current_schema()" in sql
    assert "SET search_path = pg_catalog, %I, pg_temp" in sql
    assert "SET search_path = pg_catalog, platform, pg_temp" not in sql
    assert "SET search_path = pg_catalog, public, pg_temp" not in sql
    assert function_names == _SEARCH_PATH_FUNCTIONS
    assert sql.count("ALTER FUNCTION %I.%I()") == 1


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="session")
async def test_migrated_trigger_search_paths_match_the_live_schema(
    server, setup_test_user: str
) -> None:
    await ensure_personal_org(setup_test_user)
    names_sql = ", ".join(f"'{name}'" for name in sorted(_SEARCH_PATH_FUNCTIONS))
    rows = await prisma.query_raw(
        f"""
        SELECT p.proname, p.proconfig, current_schema() AS schema_name
        FROM pg_proc AS p
        JOIN pg_namespace AS n ON n.oid = p.pronamespace
        WHERE n.nspname = current_schema()
          AND p.pronargs = 0
          AND p.proname IN ({names_sql})
        """
    )

    assert {row["proname"] for row in rows} == _SEARCH_PATH_FUNCTIONS
    schema_names = {row["schema_name"] for row in rows}
    assert len(schema_names) == 1
    expected = f"search_path=pg_catalog, {schema_names.pop()}, pg_temp"
    assert all(expected in row["proconfig"] for row in rows)

    org_members = await prisma.orgmember.count(where={"userId": setup_test_user})
    team_members = await prisma.teammember.count(where={"userId": setup_test_user})
    assert org_members >= 1
    assert team_members >= 1


def test_workspace_migration_quarantines_legacy_rows_then_defaults_new_rows() -> None:
    sql = _workspace_migration_sql()

    assert 'ADD COLUMN "scopeResolved" BOOLEAN NOT NULL DEFAULT false' in sql
    assert sql.count('ALTER COLUMN "scopeResolved" SET DEFAULT true') == 2
    assert sql.count("ALTER COLUMN \"scopeKey\" SET DEFAULT '__scope__'") == 2


def test_validation_lock_migration_removes_the_legacy_root_name_index() -> None:
    sql = _validation_lock_migration_sql()

    drop = 'DROP INDEX IF EXISTS "UserWorkspaceFolder_workspaceId_name_root_key"'
    assert drop in sql
    assert drop not in _workspace_migration_sql()


def test_workspace_migration_enforces_exact_sources_folders_and_shares() -> None:
    sql = _workspace_migration_sql()

    assert "workspace artifact session scope mismatch" in sql
    assert "workspace artifact execution scope mismatch" in sql
    assert "workspace artifact folder scope mismatch" in sql
    assert "workspace folder parent scope mismatch" in sql
    assert "shared workspace file scope mismatch" in sql
    assert sql.count("SET search_path = pg_catalog, platform, pg_temp") == 3


def test_notification_migration_enforces_listing_team_provenance() -> None:
    sql = _notification_migration_sql()

    assert 'SELECT "organizationId", "teamId" INTO graph_org_id, graph_team_id' in sql
    assert 'NEW."teamId" IS DISTINCT FROM graph_team_id' in sql
    assert (
        'UPDATE OF "agentGraphId", "agentGraphVersion", "storeListingId", '
        '"organizationId", "teamId"' in sql
    )


def test_notification_migration_allows_lifecycle_after_source_soft_delete() -> None:
    sql = _notification_migration_sql()

    assert "IF TG_OP = 'UPDATE' OR NEW.\"sourceGraphExecutionId\" IS NULL THEN" in sql
    assert "alert authorization provenance is immutable" in sql


def test_tenancy_validation_writes_share_hot_parent_locks() -> None:
    sql = _validation_lock_migration_sql()

    assert "CREATE OR REPLACE FUNCTION enforce_live_tenant_resource_owner()" in sql
    assert 'FROM "User" WHERE id = owner_user_id FOR SHARE' in sql
    assert "WHERE id = owner_org_id FOR SHARE" in sql
    assert "FOR UPDATE" not in sql
