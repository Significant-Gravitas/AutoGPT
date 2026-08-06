from unittest.mock import AsyncMock, MagicMock
from urllib.parse import parse_qs, urlparse

import pytest
from pydantic import SecretStr
from pytest_mock import MockerFixture

from backend.data.model import OAuth2Credentials
from backend.integrations.oauth.reddit import RedditOAuthHandler
from backend.integrations.providers import ProviderName


def _handler() -> RedditOAuthHandler:
    return RedditOAuthHandler(
        client_id="test-client-id",
        client_secret="test-client-secret",
        redirect_uri="https://example.com/callback",
    )


def _creds(scopes: list[str] | None = None) -> OAuth2Credentials:
    return OAuth2Credentials(
        provider=ProviderName.REDDIT,
        title=None,
        username="reddit-user",
        access_token=SecretStr("access-token-value"),
        refresh_token=SecretStr("refresh-token-value"),
        access_token_expires_at=None,
        refresh_token_expires_at=None,
        scopes=scopes or ["identity", "read", "modposts"],
    )


MODERATION_SCOPES = {"modposts", "modcontributors", "modmail", "modlog"}


def test_default_login_url_never_requests_moderation_scopes():
    """
    A plain Reddit connection (read/post workflows) must not be forced to grant
    ban/remove/modmail authority. Moderation scopes are opt-in per block via
    ``RedditCredentialsField(required_scopes=...)``.
    """
    url = _handler().get_login_url([], "state-token", None)

    query = parse_qs(urlparse(url).query)
    scopes = set(query["scope"][0].split())

    assert scopes == {
        "identity",
        "read",
        "submit",
        "edit",
        "history",
        "privatemessages",
        "flair",
    }
    assert not scopes & MODERATION_SCOPES


def test_get_login_url_requests_block_declared_scopes():
    requested = sorted(set(RedditOAuthHandler.DEFAULT_SCOPES) | {"modposts"})

    url = _handler().get_login_url(requested, "state-token", None)

    query = parse_qs(urlparse(url).query)
    assert set(query["scope"][0].split()) == set(requested)


def _mock_token_response(mocker: MockerFixture, tokens: dict) -> None:
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = tokens
    mocker.patch(
        "backend.integrations.oauth.reddit.Requests",
        return_value=MagicMock(post=AsyncMock(return_value=mock_response)),
    )


def _handler_with_mocked_username(mocker: MockerFixture) -> RedditOAuthHandler:
    handler = _handler()
    mocker.patch.object(handler, "_get_username", AsyncMock(return_value="reddit-user"))
    return handler


@pytest.mark.asyncio
async def test_exchange_code_for_tokens_persists_granted_scopes(
    mocker: MockerFixture,
):
    """
    A non-moderator authorizing a moderation block gets a narrower grant than we
    asked for. Storing the *requested* scopes would make the credential claim mod
    authority it never received, turning a clear permission problem into an opaque
    403 at block-run time.
    """
    _mock_token_response(
        mocker,
        {
            "access_token": "access-token-value",
            "refresh_token": "refresh-token-value",
            "expires_in": 3600,
            "scope": "identity read",
        },
    )

    creds = await _handler_with_mocked_username(mocker).exchange_code_for_tokens(
        code="auth-code",
        scopes=["identity", "read", "modposts"],
        code_verifier=None,
    )

    assert creds.scopes == ["identity", "read"]


@pytest.mark.asyncio
async def test_exchange_code_for_tokens_normalizes_wildcard_scope(
    mocker: MockerFixture,
):
    """Reddit returns `*` to mean "everything this app may request"."""
    _mock_token_response(
        mocker,
        {
            "access_token": "access-token-value",
            "refresh_token": "refresh-token-value",
            "expires_in": 3600,
            "scope": "*",
        },
    )

    creds = await _handler_with_mocked_username(mocker).exchange_code_for_tokens(
        code="auth-code",
        scopes=["identity", "read", "modposts"],
        code_verifier=None,
    )

    assert creds.scopes == ["identity", "read", "modposts"]


@pytest.mark.asyncio
async def test_exchange_code_for_tokens_falls_back_when_scope_missing(
    mocker: MockerFixture,
):
    _mock_token_response(
        mocker,
        {
            "access_token": "access-token-value",
            "refresh_token": "refresh-token-value",
            "expires_in": 3600,
        },
    )

    creds = await _handler_with_mocked_username(mocker).exchange_code_for_tokens(
        code="auth-code",
        scopes=["identity", "read", "modposts"],
        code_verifier=None,
    )

    assert creds.scopes == ["identity", "read", "modposts"]


@pytest.mark.asyncio
async def test_refresh_tokens_tracks_granted_scopes(mocker: MockerFixture):
    _mock_token_response(
        mocker,
        {
            "access_token": "new-access-token",
            "expires_in": 3600,
            "scope": "identity read",
        },
    )

    refreshed = await _handler_with_mocked_username(mocker)._refresh_tokens(_creds())

    assert refreshed.scopes == ["identity", "read"]
    assert refreshed.refresh_token == SecretStr("refresh-token-value")


@pytest.mark.asyncio
async def test_refresh_tokens_keeps_existing_scopes_when_absent(
    mocker: MockerFixture,
):
    _mock_token_response(
        mocker,
        {
            "access_token": "new-access-token",
            "expires_in": 3600,
        },
    )

    refreshed = await _handler_with_mocked_username(mocker)._refresh_tokens(_creds())

    assert refreshed.scopes == ["identity", "read", "modposts"]
