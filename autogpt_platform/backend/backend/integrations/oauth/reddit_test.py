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


def test_get_login_url_uses_default_scopes_including_moderation():
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
        "modposts",
        "modcontributors",
        "modmail",
        "modlog",
    }


@pytest.mark.asyncio
async def test_exchange_code_for_tokens_keeps_requested_scopes(
    mocker: MockerFixture,
):
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = {
        "access_token": "access-token-value",
        "refresh_token": "refresh-token-value",
        "expires_in": 3600,
        "scope": "identity read",
    }
    mock_post = AsyncMock(return_value=mock_response)
    mocker.patch(
        "backend.integrations.oauth.reddit.Requests",
        return_value=MagicMock(post=mock_post),
    )

    handler = _handler()
    mocker.patch.object(
        handler,
        "_get_username",
        AsyncMock(return_value="reddit-user"),
    )

    creds = await handler.exchange_code_for_tokens(
        code="auth-code",
        scopes=["identity", "read", "modposts"],
        code_verifier=None,
    )

    assert creds.scopes == ["identity", "read", "modposts"]


@pytest.mark.asyncio
async def test_refresh_tokens_keeps_existing_scopes(mocker: MockerFixture):
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = {
        "access_token": "new-access-token",
        "expires_in": 3600,
        "scope": "identity read",
    }
    mock_post = AsyncMock(return_value=mock_response)
    mocker.patch(
        "backend.integrations.oauth.reddit.Requests",
        return_value=MagicMock(post=mock_post),
    )

    handler = _handler()
    mocker.patch.object(
        handler,
        "_get_username",
        AsyncMock(return_value="reddit-user"),
    )

    refreshed = await handler._refresh_tokens(_creds())

    assert refreshed.scopes == ["identity", "read", "modposts"]
    assert refreshed.refresh_token == SecretStr("refresh-token-value")
