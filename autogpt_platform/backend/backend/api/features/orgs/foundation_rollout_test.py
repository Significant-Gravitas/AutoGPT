from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from autogpt_libs.auth.models import RequestContext
from fastapi import HTTPException

from backend.api.features.orgs import rollout, routes, team_routes
from backend.api.features.orgs.model import CreateOrgRequest
from backend.util.feature_flag import Flag


def owner_context():
    return RequestContext(
        user_id="owner",
        org_id="personal",
        team_id=None,
        is_org_owner=True,
        is_org_admin=True,
        is_org_billing_manager=False,
        is_team_admin=False,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )


@pytest.mark.asyncio
async def test_rollout_uses_literal_flag_and_fails_closed(mocker):
    check = mocker.patch.object(
        rollout, "is_feature_enabled", AsyncMock(return_value=False)
    )
    with pytest.raises(HTTPException) as error:
        await rollout.require_org_collaboration("owner")
    assert error.value.status_code == 403
    assert Flag.SHOW_ORG_SETTINGS.value == "SHOW_ORG_SETTINGS"
    check.assert_awaited_once_with(Flag.SHOW_ORG_SETTINGS, "owner", default=False)


@pytest.mark.asyncio
async def test_disabled_creation_stops_before_database_access(monkeypatch, mocker):
    monkeypatch.setenv("FORCE_FLAG_SHOW_ORG_SETTINGS", "false")
    create = mocker.patch.object(routes.org_db, "create_org", AsyncMock())
    with pytest.raises(HTTPException) as error:
        await routes.create_org(CreateOrgRequest(name="Shared", slug="shared"), "owner")
    assert error.value.status_code == 403
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_enabled_creation_keeps_existing_behavior(mocker):
    created = MagicMock()
    mocker.patch.object(routes.org_db, "create_org", AsyncMock(return_value=created))
    assert (
        await routes.create_org(CreateOrgRequest(name="Shared", slug="shared"), "owner")
        is created
    )


@pytest.mark.asyncio
async def test_disabled_lists_only_canonical_personal_organization(monkeypatch, mocker):
    monkeypatch.setenv("FORCE_FLAG_SHOW_ORG_SETTINGS", "false")
    personal = MagicMock()
    default = mocker.patch.object(
        routes.org_db,
        "get_user_default_team",
        AsyncMock(return_value=("personal", "default")),
    )
    get_org = mocker.patch.object(
        routes.org_db, "get_org", AsyncMock(return_value=personal)
    )
    roster = mocker.patch.object(routes.org_db, "list_user_orgs", AsyncMock())
    assert await routes.list_orgs("owner") == [personal]
    default.assert_awaited_once_with("owner")
    get_org.assert_awaited_once_with("personal")
    roster.assert_not_awaited()


@pytest.mark.asyncio
async def test_disabled_keeps_default_workspace_and_hides_collaboration(
    monkeypatch, mocker
):
    monkeypatch.setenv("FORCE_FLAG_SHOW_ORG_SETTINGS", "false")
    default = SimpleNamespace(id="default")
    mocker.patch.object(
        team_routes.org_db,
        "get_user_default_team",
        AsyncMock(return_value=("personal", "default")),
    )
    get_teams = mocker.patch.object(
        team_routes.team_db,
        "list_teams",
        AsyncMock(return_value=[default, SimpleNamespace(id="shared")]),
    )
    assert await team_routes.list_teams("personal", owner_context()) == [default]
    get_teams.assert_awaited_once_with("personal", "owner")
    with pytest.raises(HTTPException) as error:
        await team_routes.get_team("personal", "shared", owner_context())
    assert error.value.status_code == 403


@pytest.mark.asyncio
async def test_disabled_default_context_fails_closed_when_unavailable(
    monkeypatch, mocker
):
    monkeypatch.setenv("FORCE_FLAG_SHOW_ORG_SETTINGS", "false")
    mocker.patch.object(
        routes.org_db, "get_user_default_team", AsyncMock(return_value=(None, None))
    )
    with pytest.raises(HTTPException) as error:
        await routes.get_default_org("owner")
    assert error.value.status_code == 503
