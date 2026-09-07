from unittest.mock import AsyncMock, MagicMock

import pytest
from autogpt_libs.auth.models import RequestContext
from fastapi import HTTPException

from backend.api.features.orgs import invitation_routes, rollout, routes
from backend.api.features.orgs.model import (
    AddMemberRequest,
    CreateInvitationRequest,
    CreateOrgRequest,
)
from backend.util.feature_flag import Flag


def owner_context():
    return RequestContext(
        user_id="owner",
        org_id="org",
        team_id=None,
        is_org_owner=True,
        is_org_admin=True,
        is_org_billing_manager=False,
        is_team_admin=False,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation", ["create", "convert", "add_member", "invite", "resend", "accept"]
)
async def test_disabled_rollout_blocks_collaboration_before_database_access(
    operation, monkeypatch, mocker
):
    monkeypatch.setenv("FORCE_FLAG_SHOW_ORG_SETTINGS", "false")
    ctx = owner_context()
    blocked = AsyncMock(
        side_effect=AssertionError("Database access before rollout gate")
    )
    if operation == "create":
        mocker.patch.object(routes.org_db, "create_org", blocked)
        call = routes.create_org(
            CreateOrgRequest(name="New organization", slug="new-organization"), "owner"
        )
    elif operation == "convert":
        mocker.patch.object(routes.org_db, "convert_personal_org", blocked)
        call = routes.convert_org("org", ctx)
    elif operation == "add_member":
        mocker.patch.object(routes.org_db, "add_org_member", blocked)
        call = routes.add_member("org", AddMemberRequest(user_id="recipient"), ctx)
    elif operation == "invite":
        mocker.patch.object(invitation_routes, "_create_invitation_locked", blocked)
        call = invitation_routes.create_invitation(
            "org", CreateInvitationRequest(email="recipient@example.com"), ctx
        )
    elif operation == "resend":
        mocker.patch.object(invitation_routes, "_get_org_invitation", blocked)
        call = invitation_routes.resend_invitation("org", "invitation", ctx)
    else:
        mocker.patch.object(
            invitation_routes,
            "prisma",
            MagicMock(orginvitation=MagicMock(find_unique=blocked)),
        )
        call = invitation_routes.accept_invitation("token", "recipient")
    with pytest.raises(HTTPException) as error:
        await call
    assert error.value.status_code == 403
    assert "not enabled" in error.value.detail
    blocked.assert_not_awaited()


@pytest.mark.asyncio
async def test_disabled_rollout_keeps_existing_organization_reads_available(
    monkeypatch, mocker
):
    monkeypatch.setenv("FORCE_FLAG_SHOW_ORG_SETTINGS", "false")
    roster = mocker.patch.object(
        routes.org_db, "list_user_orgs", AsyncMock(return_value=[])
    )
    assert await routes.list_orgs("owner") == []
    roster.assert_awaited_once_with("owner")


@pytest.mark.asyncio
async def test_rollout_uses_existing_literal_flag_key_and_defaults_closed(mocker):
    check = mocker.patch.object(
        rollout, "is_feature_enabled", AsyncMock(return_value=False)
    )
    with pytest.raises(HTTPException) as error:
        await rollout.require_org_collaboration("owner")
    assert error.value.status_code == 403
    assert Flag.SHOW_ORG_SETTINGS.value == "SHOW_ORG_SETTINGS"
    check.assert_awaited_once_with(Flag.SHOW_ORG_SETTINGS, "owner", default=False)


@pytest.mark.asyncio
async def test_direct_add_requires_recipient_to_be_in_the_rollout(mocker):
    check = mocker.patch.object(
        rollout, "is_feature_enabled", AsyncMock(side_effect=[True, False])
    )
    add = mocker.patch.object(routes.org_db, "add_org_member", AsyncMock())
    with pytest.raises(HTTPException) as error:
        await routes.add_member(
            "org", AddMemberRequest(user_id="recipient"), owner_context()
        )
    assert error.value.status_code == 403
    assert [call.args[1] for call in check.await_args_list] == ["owner", "recipient"]
    add.assert_not_awaited()


@pytest.mark.asyncio
async def test_enabled_rollout_reaches_org_creation(mocker):
    created = MagicMock()
    create = mocker.patch.object(
        routes.org_db, "create_org", AsyncMock(return_value=created)
    )
    assert (
        await routes.create_org(CreateOrgRequest(name="New", slug="new-org"), "owner")
        is created
    )
    create.assert_awaited_once()


@pytest.mark.asyncio
async def test_disabled_rollout_still_allows_removing_existing_members(
    monkeypatch, mocker
):
    monkeypatch.setenv("FORCE_FLAG_SHOW_ORG_SETTINGS", "false")
    remove = mocker.patch.object(routes.org_db, "remove_org_member", AsyncMock())
    await routes.remove_member("org", "recipient", owner_context())
    remove.assert_awaited_once_with("org", "recipient", requesting_user_id="owner")
