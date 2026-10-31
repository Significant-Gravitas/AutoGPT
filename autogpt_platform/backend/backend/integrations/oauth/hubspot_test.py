import time
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import parse_qs, urlparse

import pytest
from pydantic import SecretStr
from pytest_mock import MockerFixture

from backend.data.model import OAuth2Credentials
from backend.integrations.oauth import HANDLERS_BY_NAME
from backend.integrations.oauth.hubspot import HubSpotOAuthError, HubSpotOAuthHandler
from backend.integrations.providers import ProviderName

REDIRECT_URI = "https://platform.example.com/auth/integrations/oauth_callback"
SCOPES = [
    "oauth",
    "crm.objects.contacts.read",
    "crm.objects.contacts.write",
    "crm.objects.companies.read",
    "crm.objects.companies.write",
]
TOKEN_RESPONSE = {
    "token_type": "bearer",
    "access_token": "new-access-token",
    "refresh_token": "na1-refresh-token",
    "hub_id": 1234567,
    "scopes": SCOPES,
    "expires_in": 1800,
}
INTROSPECT_RESPONSE = {
    "active": True,
    "hub_id": 1234567,
    "user": "jdoe@example.com",
    "hub_domain": "example.com",
    "scopes": SCOPES,
    "token_use": "access_token",
}


def _handler() -> HubSpotOAuthHandler:
    return HubSpotOAuthHandler(
        client_id="test-client-id",
        client_secret="test-client-secret",
        redirect_uri=REDIRECT_URI,
    )


def _response(body: dict, status: int = 200) -> MagicMock:
    response = MagicMock()
    response.ok = 200 <= status < 300
    response.status = status
    response.json.return_value = body
    return response


def _mock_requests(mocker: MockerFixture, **methods: AsyncMock) -> MagicMock:
    requests = mocker.patch(
        "backend.integrations.oauth.hubspot.Requests",
        return_value=MagicMock(**methods),
    )
    return requests


def _creds(expires_at: int | None = None) -> OAuth2Credentials:
    return OAuth2Credentials(
        id="11111111-2222-3333-4444-555555555555",
        provider=ProviderName.HUBSPOT,
        title="example.com",
        username="jdoe@example.com",
        access_token=SecretStr("old-access-token"),
        refresh_token=SecretStr("na1-refresh-token"),
        access_token_expires_at=expires_at,
        refresh_token_expires_at=None,
        scopes=SCOPES,
        metadata={"hub_id": 1234567},
    )


def test_handler_is_registered_for_hubspot():
    assert HANDLERS_BY_NAME["hubspot"] is HubSpotOAuthHandler


def test_login_url_asks_for_the_scopes_the_blocks_use():
    url = _handler().get_login_url([], "state-token", None)

    parsed = urlparse(url)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == (
        "https://app.hubspot.com/oauth/authorize"
    )
    query = parse_qs(parsed.query)
    assert query["client_id"] == ["test-client-id"]
    assert query["redirect_uri"] == [REDIRECT_URI]
    assert query["state"] == ["state-token"]
    # Space separated, and no wider than the blocks need.
    assert query["scope"][0].split(" ") == SCOPES


def test_login_url_always_includes_every_default_scope():
    # HubSpot rejects an authorize URL that leaves out a scope the app requires,
    # so a block asking for a subset must still get the full set.
    url = _handler().get_login_url(
        ["crm.objects.contacts.read", "crm.objects.owners.read"], "state", None
    )

    scopes = parse_qs(urlparse(url).query)["scope"][0].split(" ")
    assert scopes == SCOPES + ["crm.objects.owners.read"]


async def test_exchange_code_for_tokens(mocker: MockerFixture):
    post = AsyncMock(
        side_effect=[_response(TOKEN_RESPONSE), _response(INTROSPECT_RESPONSE)]
    )
    _mock_requests(mocker, post=post)

    before = int(time.time())
    creds = await _handler().exchange_code_for_tokens("auth-code", SCOPES, None)

    token_call, introspect_call = post.await_args_list
    assert token_call.args[0] == "https://api.hubspot.com/oauth/2026-09/token"
    assert token_call.kwargs["data"] == {
        "client_id": "test-client-id",
        "client_secret": "test-client-secret",
        "grant_type": "authorization_code",
        "code": "auth-code",
        "redirect_uri": REDIRECT_URI,
    }
    assert token_call.kwargs["allow_redirects"] is False
    assert introspect_call.args[0] == (
        "https://api.hubspot.com/oauth/2026-09/token/introspect"
    )
    assert introspect_call.kwargs["data"] == {
        "client_id": "test-client-id",
        "client_secret": "test-client-secret",
        "token_type_hint": "access_token",
        "token": "new-access-token",
    }
    assert introspect_call.kwargs["allow_redirects"] is False

    assert creds.provider == ProviderName.HUBSPOT
    assert creds.access_token.get_secret_value() == "new-access-token"
    assert creds.refresh_token is not None
    assert creds.refresh_token.get_secret_value() == "na1-refresh-token"
    assert creds.access_token_expires_at is not None
    assert before + 1800 <= creds.access_token_expires_at <= int(time.time()) + 1800
    assert creds.scopes == SCOPES
    assert creds.username == "jdoe@example.com (portal 1234567)"
    assert creds.title == "example.com"
    assert creds.metadata == {"hub_id": 1234567}


async def test_exchange_survives_a_failed_introspection(mocker: MockerFixture):
    post = AsyncMock(
        side_effect=[_response(TOKEN_RESPONSE), _response({"error": "x"}, 500)]
    )
    _mock_requests(mocker, post=post)

    creds = await _handler().exchange_code_for_tokens("auth-code", SCOPES, None)

    assert creds.access_token.get_secret_value() == "new-access-token"
    assert creds.username is None
    assert creds.metadata == {"hub_id": 1234567}


async def test_same_user_in_two_portals_gets_two_usernames(mocker: MockerFixture):
    # The callback merges sign-ins with the same username into one credential,
    # so a second portal must not share the first one's username.
    other_portal = {**TOKEN_RESPONSE, "hub_id": 7654321}
    post = AsyncMock(
        side_effect=[
            _response(TOKEN_RESPONSE),
            _response(INTROSPECT_RESPONSE),
            _response(other_portal),
            _response({**INTROSPECT_RESPONSE, "hub_id": 7654321}),
        ]
    )
    _mock_requests(mocker, post=post)

    first = await _handler().exchange_code_for_tokens("code-1", SCOPES, None)
    second = await _handler().exchange_code_for_tokens("code-2", SCOPES, None)

    assert first.username != second.username
    assert second.metadata == {"hub_id": 7654321}


async def test_exchange_raises_hubspots_error(mocker: MockerFixture):
    error = {
        "error": "invalid_grant",
        "error_description": "authorization code is invalid",
    }
    _mock_requests(mocker, post=AsyncMock(return_value=_response(error, 400)))

    with pytest.raises(HubSpotOAuthError, match="authorization code is invalid"):
        await _handler().exchange_code_for_tokens("bad-code", SCOPES, None)


async def test_expired_access_token_is_refreshed(mocker: MockerFixture):
    refreshed = {**TOKEN_RESPONSE, "access_token": "refreshed-access-token"}
    post = AsyncMock(return_value=_response(refreshed))
    _mock_requests(mocker, post=post)
    handler = _handler()
    # 30-minute tokens: one issued 29 minutes ago is inside the refresh window.
    creds = _creds(expires_at=int(time.time()) + 60)

    assert handler.needs_refresh(creds)
    token = await handler.get_access_token(creds)

    assert token == "refreshed-access-token"
    post.assert_awaited_once()
    assert post.await_args.kwargs["data"] == {
        "client_id": "test-client-id",
        "client_secret": "test-client-secret",
        "grant_type": "refresh_token",
        "refresh_token": "na1-refresh-token",
    }
    assert post.await_args.kwargs["allow_redirects"] is False


async def test_refresh_keeps_the_credential_identity(mocker: MockerFixture):
    refreshed = {**TOKEN_RESPONSE, "access_token": "refreshed-access-token"}
    del refreshed["refresh_token"]
    _mock_requests(mocker, post=AsyncMock(return_value=_response(refreshed)))
    old = _creds(expires_at=int(time.time()) - 10)

    new = await _handler().refresh_tokens(old)

    assert new.id == old.id
    assert new.title == old.title
    assert new.username == old.username
    assert new.metadata == old.metadata
    assert new.access_token.get_secret_value() == "refreshed-access-token"
    # HubSpot left the refresh token out: the old one still works.
    assert new.refresh_token is not None
    assert new.refresh_token.get_secret_value() == "na1-refresh-token"
    assert new.access_token_expires_at is not None
    assert new.access_token_expires_at > int(time.time()) + 1700


async def test_fresh_access_token_is_not_refreshed(mocker: MockerFixture):
    post = AsyncMock()
    _mock_requests(mocker, post=post)

    token = await _handler().get_access_token(
        _creds(expires_at=int(time.time()) + 1500)
    )

    assert token == "old-access-token"
    post.assert_not_awaited()


async def test_refresh_raises_when_hubspot_refuses(mocker: MockerFixture):
    error = {
        "error": "invalid_grant",
        "error_description": "refresh token is invalid, expired or revoked",
    }
    _mock_requests(mocker, post=AsyncMock(return_value=_response(error, 400)))

    with pytest.raises(HubSpotOAuthError, match="expired or revoked"):
        await _handler().refresh_tokens(_creds(expires_at=0))


async def test_revoke_revokes_the_refresh_token(mocker: MockerFixture):
    post = AsyncMock(return_value=_response({}, 204))
    _mock_requests(mocker, post=post)

    assert await _handler().revoke_tokens(_creds()) is True

    post.assert_awaited_once()
    # The token goes in the form body: HubSpot rejects tokens in the URL, and
    # a URL ends up in logs.
    assert post.await_args.args[0] == (
        "https://api.hubapi.com/oauth/2026-09/token/revoke"
    )
    assert post.await_args.kwargs["data"] == {
        "client_id": "test-client-id",
        "client_secret": "test-client-secret",
        "token_type_hint": "refresh_token",
        "token": "na1-refresh-token",
    }
    assert post.await_args.kwargs["allow_redirects"] is False


async def test_failed_revoke_does_not_raise(mocker: MockerFixture):
    _mock_requests(mocker, post=AsyncMock(return_value=_response({}, 400)))

    assert await _handler().revoke_tokens(_creds()) is False


async def test_every_hubspot_request_has_a_retry_cap(mocker: MockerFixture):
    # Uncapped, a throttled or failing HubSpot is retried forever.
    post = AsyncMock(
        side_effect=[
            _response(TOKEN_RESPONSE),
            _response(INTROSPECT_RESPONSE),
            _response({}, 204),
        ]
    )
    requests = _mock_requests(mocker, post=post)

    await _handler().exchange_code_for_tokens("auth-code", SCOPES, None)
    await _handler().revoke_tokens(_creds())

    assert requests.call_count == 3
    for call in requests.call_args_list:
        assert call.kwargs == {"raise_for_status": False, "retry_max_attempts": 3}


async def test_revoke_survives_hubspot_being_unreachable(mocker: MockerFixture):
    _mock_requests(mocker, post=AsyncMock(side_effect=ConnectionError("down")))

    assert await _handler().revoke_tokens(_creds()) is False
