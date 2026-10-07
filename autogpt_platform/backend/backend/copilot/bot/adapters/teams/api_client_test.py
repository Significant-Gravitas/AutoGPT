"""Tests for the Bot Connector client: token minting and the outbound gate."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.bot.adapters.teams import config
from backend.copilot.bot.adapters.teams.api_client import (
    _TOKEN_REFRESH_MARGIN_SECONDS,
    TeamsApiError,
    TeamsClient,
)

_CONFIG_PATH = "backend.copilot.bot.adapters.teams.config"
_ALLOWED = "https://smba.trafficmanager.net/emea/"


def _token_response(status_code=200, **body):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = {"access_token": "tok", "expires_in": 3600, **body}
    response.text = "boom"
    return response


def _creds():
    return (
        patch(f"{_CONFIG_PATH}.get_app_id", return_value="app-id"),
        patch(f"{_CONFIG_PATH}.get_app_password", return_value="secret"),
        patch(f"{_CONFIG_PATH}.get_tenant_id", return_value="tenant-id"),
        patch(f"{_CONFIG_PATH}.allow_unverified_requests", return_value=False),
    )


@pytest.mark.asyncio
async def test_token_is_minted_against_the_bots_own_tenant():
    """The tenant-less botframework.com authority is the multi-tenant form.

    Microsoft only issues single-tenant registrations now, and using the wrong
    authority 401s every outbound call — the PR's most-cited failure mode.
    """
    client = TeamsClient()
    client._http = MagicMock()
    client._http.post = AsyncMock(return_value=_token_response())
    app_id, password, tenant, unverified = _creds()
    with app_id, password, tenant, unverified:
        token = await client._access_token()

    assert token == "tok"
    url = client._http.post.await_args.args[0]
    assert url == ("https://login.microsoftonline.com/tenant-id/oauth2/v2.0/token")
    body = client._http.post.await_args.kwargs["data"]
    assert body["grant_type"] == "client_credentials"
    assert body["client_id"] == "app-id"
    assert body["client_secret"] == "secret"
    assert body["scope"] == config.CONNECTOR_SCOPE


@pytest.mark.asyncio
async def test_token_expiry_keeps_a_refresh_margin():
    client = TeamsClient()
    client._http = MagicMock()
    client._http.post = AsyncMock(return_value=_token_response(expires_in=3600))
    app_id, password, tenant, unverified = _creds()
    with app_id, password, tenant, unverified:
        before = time.monotonic()
        await client._access_token()

    # Expire early so a call never rides a token that dies mid-flight.
    assert client._token_expires_at <= before + 3600 - _TOKEN_REFRESH_MARGIN_SECONDS + 1


@pytest.mark.asyncio
async def test_a_short_lived_token_never_expires_in_the_past():
    # A lifetime under the margin would otherwise compute a already-expired
    # deadline and remint on every single call.
    client = TeamsClient()
    client._http = MagicMock()
    client._http.post = AsyncMock(return_value=_token_response(expires_in=60))
    app_id, password, tenant, unverified = _creds()
    with app_id, password, tenant, unverified:
        await client._access_token()

    assert client._token_expires_at > time.monotonic()


@pytest.mark.asyncio
async def test_a_cached_token_is_not_reminted():
    client = TeamsClient()
    client._http = MagicMock()
    client._http.post = AsyncMock(return_value=_token_response())
    app_id, password, tenant, unverified = _creds()
    with app_id, password, tenant, unverified:
        await client._access_token()
        await client._access_token()

    assert client._http.post.await_count == 1


@pytest.mark.asyncio
async def test_a_token_response_without_a_token_is_an_error():
    client = TeamsClient()
    client._http = MagicMock()
    client._http.post = AsyncMock(return_value=_token_response(access_token=""))
    app_id, password, tenant, unverified = _creds()
    with app_id, password, tenant, unverified, pytest.raises(TeamsApiError):
        await client._access_token()


@pytest.mark.asyncio
async def test_a_rejected_grant_is_an_error():
    client = TeamsClient()
    client._http = MagicMock()
    client._http.post = AsyncMock(return_value=_token_response(status_code=401))
    app_id, password, tenant, unverified = _creds()
    with app_id, password, tenant, unverified, pytest.raises(TeamsApiError):
        await client._access_token()


@pytest.mark.asyncio
async def test_an_untrusted_service_url_is_refused_before_any_request():
    """The allowlist is the outbound gate, so nothing may be dialed first.

    serviceUrl arrives on the activity body, which the Connector token does
    not sign — sending the bearer anywhere it names would hand out the token.
    """
    client = TeamsClient()
    client._http = MagicMock()
    client._http.request = AsyncMock()
    app_id, password, tenant, unverified = _creds()
    with app_id, password, tenant, unverified, pytest.raises(TeamsApiError):
        await client.send_activity(
            "https://attacker.example/", "19:conv", {"type": "message"}
        )

    client._http.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_allowed_service_url_is_dialed_with_the_bearer():
    client = TeamsClient()
    response = MagicMock()
    response.status_code = 200
    response.content = b"{}"
    response.json.return_value = {}
    client._http = MagicMock()
    client._http.request = AsyncMock(return_value=response)
    client._token = "tok"
    client._token_expires_at = time.monotonic() + 600
    app_id, password, tenant, unverified = _creds()
    with app_id, password, tenant, unverified:
        await client.send_activity(_ALLOWED, "19:conv", {"type": "message"})

    method, url = client._http.request.await_args.args
    assert method == "POST"
    assert url == f"{_ALLOWED.rstrip('/')}/v3/conversations/19:conv/activities"
    headers = client._http.request.await_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer tok"


@pytest.mark.asyncio
async def test_the_playground_sends_no_bearer():
    # Nothing issues or checks tokens in that mode, and there are no
    # credentials to mint one from.
    client = TeamsClient()
    with patch(f"{_CONFIG_PATH}.allow_unverified_requests", return_value=True):
        assert await client.bearer_headers() == {}


def _ok_response():
    response = MagicMock()
    response.status_code = 200
    response.content = b"{}"
    response.json.return_value = {}
    return response


def _authed_client() -> TeamsClient:
    client = TeamsClient()
    client._http = MagicMock()
    client._token = "tok"
    client._token_expires_at = time.monotonic() + 600
    return client


@pytest.mark.asyncio
async def test_update_activity_builds_the_expected_url():
    client = _authed_client()
    client._http.request = AsyncMock(return_value=_ok_response())
    app_id, password, tenant, unverified = _creds()
    with app_id, password, tenant, unverified:
        await client.update_activity(_ALLOWED, "19:conv", "act-1", {"type": "message"})

    method, url = client._http.request.await_args.args
    assert method == "PUT"
    assert url == f"{_ALLOWED.rstrip('/')}/v3/conversations/19:conv/activities/act-1"


@pytest.mark.asyncio
async def test_update_activity_encodes_a_slash_in_activity_id():
    # activity_id is an AutoPilot-tool-supplied ref_id with no id grammar of
    # its own — a raw "/" must not be able to splice in an extra path segment
    # and redirect the PUT to a different (unauthorized) conversation.
    client = _authed_client()
    client._http.request = AsyncMock(return_value=_ok_response())
    app_id, password, tenant, unverified = _creds()
    with app_id, password, tenant, unverified:
        await client.update_activity(
            _ALLOWED, "19:conv", "../../conversations/other/activities/x", {}
        )

    _, url = client._http.request.await_args.args
    # The malicious segment is fully percent-encoded, so it stays one inert
    # path component instead of being normalized into a different path.
    assert url.startswith(
        f"{_ALLOWED.rstrip('/')}/v3/conversations/19:conv/activities/"
    )
    assert "/conversations/other/" not in url
    assert "%2F" in url


@pytest.mark.asyncio
@pytest.mark.parametrize("activity_id", [".", ".."])
async def test_update_activity_rejects_bare_dot_segments(activity_id):
    client = _authed_client()
    client._http.request = AsyncMock(return_value=_ok_response())
    app_id, password, tenant, unverified = _creds()
    with app_id, password, tenant, unverified, pytest.raises(TeamsApiError):
        await client.update_activity(_ALLOWED, "19:conv", activity_id, {})

    client._http.request.assert_not_awaited()
