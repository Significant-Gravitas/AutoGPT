from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from backend.api.features.orgs import (
    grant_routes,
    invitation_routes,
    memory_routes,
    routes,
    team_routes,
)
from backend.api.features.orgs.grant_model import CreateGrantRequest
from backend.api.features.orgs.model import UpdateOrgRequest
from backend.api.features.orgs.rollout_test import owner_context
from backend.api.features.orgs.team_model import (
    AddTeamMemberRequest,
    CreateTeamRequest,
    UpdateTeamMemberRequest,
    UpdateTeamRequest,
)
from backend.api.features.transfers import routes as transfer_routes
from backend.api.features.transfers.model import CreateTransferRequest


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    [
        "org_details",
        "org_update",
        "org_members",
        "org_spend",
        "org_aliases",
        "team_create",
        "team_details",
        "team_update",
        "team_join",
        "team_members",
        "team_add_member",
        "team_update_member",
        "grants_create",
        "grants_list",
        "grants_received",
        "memory_held",
        "memory_active",
        "memory_approve",
        "invitations_list",
        "invitations_pending",
    ],
)
async def test_flag_off_blocks_collaboration_surfaces_before_data_access(
    operation, monkeypatch, mocker
):
    monkeypatch.setenv("FORCE_FLAG_SHOW_ORG_SETTINGS", "false")
    ctx = owner_context()
    blocked = AsyncMock(side_effect=AssertionError("Collaboration data was accessed"))
    for module, methods in (
        (
            routes.org_db,
            ("get_org", "update_org", "list_org_members", "list_org_aliases"),
        ),
        (
            team_routes.team_db,
            (
                "create_team",
                "get_team_for_viewer",
                "update_team",
                "join_team",
                "list_team_members",
                "add_team_member",
                "update_team_member",
            ),
        ),
        (
            grant_routes.grant_db,
            ("upsert_grant", "list_grants_for_graph", "list_received_grants"),
        ),
        (
            memory_routes.memory_db,
            ("list_held_memories", "list_active_memories", "approve_held_memory"),
        ),
    ):
        for method in methods:
            mocker.patch.object(module, method, blocked)
    mocker.patch.object(routes, "get_org_spend_by_team", blocked)
    mocker.patch.object(
        invitation_routes,
        "prisma",
        MagicMock(orginvitation=MagicMock(find_many=blocked)),
    )

    calls = {
        "org_details": lambda: routes.get_org("org", ctx),
        "org_update": lambda: routes.update_org(
            "org", UpdateOrgRequest(name="New"), ctx
        ),
        "org_members": lambda: routes.list_members("org", ctx),
        "org_spend": lambda: routes.get_org_spend("org", ctx),
        "org_aliases": lambda: routes.list_aliases("org", ctx),
        "team_create": lambda: team_routes.create_team(
            "org", CreateTeamRequest(name="New team"), ctx
        ),
        "team_details": lambda: team_routes.get_team("org", "team", ctx),
        "team_update": lambda: team_routes.update_team(
            "org", "team", UpdateTeamRequest(name="Renamed"), ctx
        ),
        "team_join": lambda: team_routes.join_team("org", "team", ctx),
        "team_members": lambda: team_routes.list_members("org", "team", ctx),
        "team_add_member": lambda: team_routes.add_member(
            "org", "team", AddTeamMemberRequest(user_id="recipient"), ctx
        ),
        "team_update_member": lambda: team_routes.update_member(
            "org", "team", "recipient", UpdateTeamMemberRequest(is_admin=True), ctx
        ),
        "grants_create": lambda: grant_routes.create_grant(
            "org", "graph", CreateGrantRequest(principal_id="team"), ctx
        ),
        "grants_list": lambda: grant_routes.list_grants("org", "graph", ctx),
        "grants_received": lambda: grant_routes.list_received_grants("org", ctx),
        "memory_held": lambda: memory_routes.list_held_memories("org", ctx),
        "memory_active": lambda: memory_routes.list_active_memories("org", ctx),
        "memory_approve": lambda: memory_routes.approve_held_memory(
            "org", "memory", ctx
        ),
        "invitations_list": lambda: invitation_routes.list_invitations("org", ctx),
        "invitations_pending": lambda: invitation_routes.list_pending_for_user("owner"),
    }
    with pytest.raises(HTTPException) as error:
        await calls[operation]()
    assert error.value.status_code == 403
    assert "not enabled" in error.value.detail
    blocked.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", ["true", "false"])
async def test_default_context_uses_canonical_owner_lookup(
    enabled, monkeypatch, mocker
):
    monkeypatch.setenv("FORCE_FLAG_SHOW_ORG_SETTINGS", enabled)
    personal = MagicMock(id="my-personal-org")
    default = mocker.patch.object(
        routes.org_db,
        "get_user_default_team",
        AsyncMock(return_value=(personal.id, "default-team")),
    )
    get_org = mocker.patch.object(
        routes.org_db, "get_org", AsyncMock(return_value=personal)
    )
    roster = mocker.patch.object(routes.org_db, "list_user_orgs", AsyncMock())
    assert await routes.get_default_org("owner") == personal
    default.assert_awaited_once_with("owner")
    get_org.assert_awaited_once_with(personal.id)
    roster.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_personal_context_does_not_fall_back_to_shared_org(mocker):
    mocker.patch.object(
        routes.org_db, "get_user_default_team", AsyncMock(return_value=(None, None))
    )
    read = mocker.patch.object(routes.org_db, "get_org", AsyncMock())
    with pytest.raises(HTTPException) as error:
        await routes.get_default_org("owner")
    assert error.value.status_code == 503
    read.assert_not_awaited()


@pytest.mark.asyncio
async def test_flag_off_team_context_contains_only_owned_default_team(
    monkeypatch, mocker
):
    monkeypatch.setenv("FORCE_FLAG_SHOW_ORG_SETTINGS", "false")
    ctx = owner_context()
    default_team = MagicMock(id="default-team")
    custom_team = MagicMock(id="custom-team")
    mocker.patch.object(
        routes.org_db,
        "get_user_default_team",
        AsyncMock(return_value=("org", "default-team")),
    )
    mocker.patch.object(
        team_routes.team_db,
        "list_teams",
        AsyncMock(return_value=[custom_team, default_team]),
    )
    assert await team_routes.list_teams("org", ctx) == [default_team]


@pytest.mark.asyncio
async def test_flag_off_rejects_shared_org_context_even_for_members(
    monkeypatch, mocker
):
    monkeypatch.setenv("FORCE_FLAG_SHOW_ORG_SETTINGS", "false")
    mocker.patch.object(
        routes.org_db,
        "get_user_default_team",
        AsyncMock(return_value=("my-personal-org", "default-team")),
    )
    read = mocker.patch.object(team_routes.team_db, "list_teams", AsyncMock())
    with pytest.raises(HTTPException) as error:
        await team_routes.list_teams("org", owner_context())
    assert error.value.status_code == 403
    read.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", ["true", "false"])
@pytest.mark.parametrize("operation", ["create", "list", "approve", "execute"])
async def test_transfers_require_collaboration(enabled, operation, monkeypatch, mocker):
    monkeypatch.setenv("FORCE_FLAG_SHOW_ORG_SETTINGS", enabled)
    ctx = owner_context()
    method = "list_transfers" if operation == "list" else f"{operation}_transfer"
    result = MagicMock()
    database = mocker.patch.object(
        transfer_routes.transfer_db, method, AsyncMock(return_value=result)
    )
    calls = {
        "create": lambda: transfer_routes.create_transfer(
            CreateTransferRequest(
                resource_type="AgentGraph",
                resource_id="graph",
                target_organization_id="shared-target",
            ),
            ctx,
        ),
        "list": lambda: transfer_routes.list_transfers(ctx),
        "approve": lambda: transfer_routes.approve_transfer("transfer", ctx),
        "execute": lambda: transfer_routes.execute_transfer("transfer", ctx),
    }
    if enabled == "true":
        assert await calls[operation]() == result
        database.assert_awaited_once()
    else:
        with pytest.raises(HTTPException) as error:
            await calls[operation]()
        assert error.value.status_code == 403
        database.assert_not_awaited()


@pytest.mark.asyncio
async def test_transfer_rejection_stays_available_when_disabled(monkeypatch, mocker):
    monkeypatch.setenv("FORCE_FLAG_SHOW_ORG_SETTINGS", "false")
    reject = mocker.patch.object(
        transfer_routes.transfer_db, "reject_transfer", AsyncMock()
    )
    ctx = owner_context()
    assert await transfer_routes.reject_transfer("transfer", ctx) == reject.return_value
    reject.assert_awaited_once_with(
        transfer_id="transfer", user_id=ctx.user_id, org_id=ctx.org_id
    )
