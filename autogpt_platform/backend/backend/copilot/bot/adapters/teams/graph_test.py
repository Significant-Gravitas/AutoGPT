"""Tests for the Graph team-owner lookup behind Teams /setup."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .graph import GRAPH_SCOPE, TeamsGraphClient, TeamsGraphError

_CONFIG_PATH = "backend.copilot.bot.adapters.teams.config"
_GROUP = "6f1c2a3b-0000-4000-8000-000000000001"
_USER = "0b6c3c1e-0000-4000-8000-000000000002"
_OTHER_USER = "0b6c3c1e-0000-4000-8000-000000000003"


def _response(status_code: int = 200, body: dict | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = body or {}
    response.text = "Forbidden"
    return response


def _client(response: MagicMock) -> TeamsGraphClient:
    """A client holding a live Graph token, so only the lookup goes out."""
    client = TeamsGraphClient()
    client._http = MagicMock()
    client._http.get = AsyncMock(return_value=response)
    client._token = "graph-token"
    client._token_expires_at = time.monotonic() + 600
    return client


@pytest.mark.asyncio
async def test_an_owner_is_found_by_their_entra_id():
    client = _client(_response(body={"value": [{"userId": _USER, "roles": ["owner"]}]}))
    assert await client.is_team_owner(_GROUP, _USER) is True

    args, kwargs = client._http.get.await_args
    # "(microsoft.graph.aadUserConversationMember/userId eq '<user>')", with
    # spaces sent as %20, not the form-style "+" httpx's params= would use.
    expected_url = (
        f"https://graph.microsoft.com/v1.0/teams/{_GROUP}/members?$filter="
        "%28microsoft.graph.aadUserConversationMember%2FuserId%20eq%20"
        f"%27{_USER}%27%29"
    )
    assert args == (expected_url,)
    assert "params" not in kwargs
    assert kwargs["headers"] == {"Authorization": "Bearer graph-token"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "members",
    [[], [{"userId": _USER, "roles": []}], [{"userId": _USER, "roles": ["guest"]}]],
    ids=["not-in-team", "member", "guest"],
)
async def test_members_and_guests_are_not_owners(members):
    client = _client(_response(body={"value": members}))
    assert await client.is_team_owner(_GROUP, _USER) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "members",
    [
        [{"userId": _OTHER_USER, "roles": ["owner"]}],
        [{"userId": _USER, "roles": []}, {"userId": _OTHER_USER, "roles": ["owner"]}],
        [
            {"userId": _USER, "roles": ["guest"]},
            {"userId": _OTHER_USER, "roles": ["owner"]},
        ],
    ],
    ids=["another-owner-alone", "beside-a-member", "beside-a-guest"],
)
async def test_another_users_owner_record_makes_no_one_else_an_owner(members):
    # Should Graph ever answer with more than the member the filter names
    client = _client(_response(body={"value": members}))
    assert await client.is_team_owner(_GROUP, _USER) is False


@pytest.mark.asyncio
async def test_the_owners_record_is_matched_whatever_the_case_of_its_id():
    record = {"userId": _USER.upper(), "roles": ["Owner"]}
    client = _client(_response(body={"value": [record]}))
    assert await client.is_team_owner(_GROUP, _USER) is True


@pytest.mark.asyncio
async def test_a_refused_lookup_is_an_error_not_an_answer():
    # 403 is what Graph says when the app holds no TeamMember permission.
    client = _client(_response(status_code=403))
    with pytest.raises(TeamsGraphError):
        await client.is_team_owner(_GROUP, _USER)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "group,user",
    [
        ("19:team@thread.tacv2", _USER),
        (_GROUP, f"{_USER}') or (userId ne '"),
        (_GROUP, f"{_USER}\n"),
    ],
    ids=["thread-id", "filter-injection", "trailing-newline"],
)
async def test_ids_that_are_not_entra_object_ids_never_reach_graph(group, user):
    client = _client(_response(body={"value": [{"roles": ["owner"]}]}))
    with pytest.raises(TeamsGraphError):
        await client.is_team_owner(group, user)
    client._http.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_app_token_is_minted_for_graph_in_the_bots_tenant():
    client = TeamsGraphClient()
    client._http = MagicMock()
    token = _response(body={"access_token": "graph-token", "expires_in": 3600})
    client._http.post = AsyncMock(return_value=token)
    client._http.get = AsyncMock(return_value=_response(body={"value": []}))
    with (
        patch(f"{_CONFIG_PATH}.get_app_id", return_value="app-id"),
        patch(f"{_CONFIG_PATH}.get_app_password", return_value="secret"),
        patch(f"{_CONFIG_PATH}.get_tenant_id", return_value="tenant-id"),
    ):
        await client.is_team_owner(_GROUP, _USER)
        await client.is_team_owner(_GROUP, _USER)

    client._http.post.assert_awaited_once()
    url = client._http.post.await_args.args[0]
    assert url == "https://login.microsoftonline.com/tenant-id/oauth2/v2.0/token"
    assert client._http.post.await_args.kwargs["data"]["scope"] == GRAPH_SCOPE
