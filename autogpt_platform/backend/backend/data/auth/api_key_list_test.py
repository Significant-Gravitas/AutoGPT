"""Visibility tests for API key listing."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from prisma.models import APIKey as PrismaAPIKey

from backend.data.auth import api_key
from backend.util.exceptions import NotAuthorizedError


@pytest.fixture
def mock_key_client(mocker):
    client = AsyncMock()
    client.find_many.return_value = []
    mocker.patch.object(PrismaAPIKey, "prisma", return_value=client)
    return client


@pytest.mark.asyncio
async def test_list_keys_without_org_is_own_only(mock_key_client):
    await api_key.list_user_api_keys("u-1")

    where = mock_key_client.find_many.call_args.kwargs["where"]
    assert where == {"userId": "u-1"}


@pytest.mark.asyncio
async def test_list_keys_org_mode_is_own_exact_org(mock_key_client):
    await api_key.list_user_api_keys("u-1", organization_id="org-1")

    where = mock_key_client.find_many.call_args.kwargs["where"]
    assert where == {
        "userId": "u-1",
        "organizationId": "org-1",
    }


@pytest.mark.asyncio
async def test_list_keys_org_home_includes_only_member_teams(mock_key_client):
    await api_key.list_user_api_keys(
        "u-1", organization_id="org-1", team_ids=["team-a", "team-b"]
    )

    where = mock_key_client.find_many.call_args.kwargs["where"]
    assert where == {
        "userId": "u-1",
        "organizationId": "org-1",
        "OR": [
            {"teamIdRestriction": None},
            {"teamIdRestriction": {"in": ["team-a", "team-b"]}},
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("team_id", [None, "team-1"])
async def test_list_keys_exact_scope_distinguishes_org_home_and_team(
    mock_key_client, team_id: str | None
):
    await api_key.list_user_api_keys(
        "u-1",
        organization_id="org-1",
        team_id_restriction=team_id,
        exact_scope=True,
    )

    where = mock_key_client.find_many.call_args.kwargs["where"]
    assert where == {
        "userId": "u-1",
        "organizationId": "org-1",
        "teamIdRestriction": team_id,
    }


@pytest.mark.asyncio
async def test_get_key_org_mode_is_exact(mock_key_client):
    mock_key_client.find_first.return_value = None

    assert (
        await api_key.get_api_key_by_id("key-1", "u-1", organization_id="org-1") is None
    )

    mock_key_client.find_first.assert_awaited_once_with(
        where={"id": "key-1", "userId": "u-1", "organizationId": "org-1"}
    )


@pytest.mark.asyncio
async def test_get_key_exact_org_home_excludes_team_key(mock_key_client):
    mock_key_client.find_first.return_value = None

    assert (
        await api_key.get_api_key_by_id(
            "key-1",
            "u-1",
            organization_id="org-1",
            team_id_restriction=None,
            exact_scope=True,
        )
        is None
    )

    mock_key_client.find_first.assert_awaited_once_with(
        where={
            "id": "key-1",
            "userId": "u-1",
            "organizationId": "org-1",
            "teamIdRestriction": None,
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["revoke", "suspend", "update"])
async def test_org_management_rejects_legacy_unscoped_key(
    mock_key_client, operation: str
):
    mock_key_client.find_unique.return_value = SimpleNamespace(
        id="key-1", userId="u-1", organizationId=None
    )

    with pytest.raises(NotAuthorizedError):
        if operation == "revoke":
            await api_key.revoke_api_key("key-1", "u-1", "org-1")
        elif operation == "suspend":
            await api_key.suspend_api_key("key-1", "u-1", "org-1")
        else:
            await api_key.update_api_key_permissions("key-1", "u-1", [], "org-1")

    mock_key_client.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_team_restricted_key_stamps_resource_team(mock_key_client, mocker):
    generated = SimpleNamespace(
        head="agpt_head",
        tail="tail",
        hash="hash",
        salt="salt",
        key="agpt_plaintext",
    )
    mocker.patch.object(api_key.keysmith, "generate_key", return_value=generated)
    mock_key_client.create.side_effect = lambda *, data: SimpleNamespace(
        **data,
        status="ACTIVE",
        createdAt=datetime.now(timezone.utc),
        lastUsedAt=None,
        revokedAt=None,
        ownerType=data.get("ownerType"),
    )

    await api_key.create_api_key(
        "key",
        "u-1",
        [],
        organization_id="org-1",
        team_id_restriction="team-1",
    )

    data = mock_key_client.create.await_args.kwargs["data"]
    assert data["organizationId"] == "org-1"
    assert data["teamId"] == "team-1"
    assert data["teamIdRestriction"] == "team-1"
