"""Regression tests for GoogleOAuthHandler.

Pins the fix for SECRT-2267 / AUTOGPT-SERVER-6HB: calling revoke_tokens
with our platform OAuth2Credentials must not crash with
``AttributeError: 'OAuth2Credentials' object has no attribute
'before_request'``. The old code handed our Pydantic model to
google-auth's AuthorizedSession, which expected a google-auth
Credentials and then called .before_request() on it.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr
from pytest_mock import MockerFixture

from backend.data.model import OAuth2Credentials
from backend.integrations.oauth.google import GoogleOAuthHandler
from backend.integrations.providers import ProviderName


def _handler() -> GoogleOAuthHandler:
    return GoogleOAuthHandler(
        client_id="test-client-id",
        client_secret="test-client-secret",
        redirect_uri="https://example.com/callback",
    )


def _creds() -> OAuth2Credentials:
    return OAuth2Credentials(
        provider=ProviderName.GOOGLE,
        title=None,
        username="user@example.com",
        access_token=SecretStr("access-token-value"),
        refresh_token=SecretStr("refresh-token-value"),
        access_token_expires_at=None,
        refresh_token_expires_at=None,
        scopes=["https://www.googleapis.com/auth/userinfo.email"],
    )


@pytest.mark.asyncio
async def test_revoke_tokens_does_not_call_before_request(mocker: MockerFixture):
    """revoke_tokens must not pass our Pydantic credentials into
    google-auth's AuthorizedSession path. It should POST the token via
    the platform's async Requests helper instead."""
    mock_response = MagicMock()
    mock_response.ok = True
    mock_post = AsyncMock(return_value=mock_response)
    mocker.patch(
        "backend.integrations.oauth.google.Requests",
        return_value=MagicMock(post=mock_post),
    )

    result = await _handler().revoke_tokens(_creds())

    assert result is True
    mock_post.assert_awaited_once()
    args, kwargs = mock_post.call_args
    # URL is positional or first kwarg
    url = args[0] if args else kwargs.get("url")
    assert url == "https://oauth2.googleapis.com/revoke"
    assert kwargs["data"] == {"token": "access-token-value"}
    assert kwargs["headers"]["Content-Type"] == "application/x-www-form-urlencoded"


@pytest.mark.asyncio
async def test_revoke_tokens_returns_false_when_no_access_token(
    mocker: MockerFixture,
):
    """If the stored credential somehow lacks an access token, don't
    crash — return False so the caller can surface a clean deletion
    outcome."""
    mock_post = AsyncMock()
    mocker.patch(
        "backend.integrations.oauth.google.Requests",
        return_value=MagicMock(post=mock_post),
    )

    creds = _creds()
    # Simulate a credential row that lost its access token
    object.__setattr__(creds, "access_token", None)

    result = await _handler().revoke_tokens(creds)

    assert result is False
    mock_post.assert_not_awaited()


@pytest.mark.asyncio
async def test_revoke_tokens_propagates_http_failure(mocker: MockerFixture):
    """A non-2xx response from Google's revoke endpoint should surface
    as False, not True. Deletion callers use this signal to decide
    whether revocation was actually confirmed upstream."""
    mock_response = MagicMock()
    mock_response.ok = False
    mock_post = AsyncMock(return_value=mock_response)
    mocker.patch(
        "backend.integrations.oauth.google.Requests",
        return_value=MagicMock(post=mock_post),
    )

    result = await _handler().revoke_tokens(_creds())

    assert result is False


@pytest.mark.asyncio
async def test_revoke_tokens_uses_bounded_retries(mocker: MockerFixture):
    """Cursor Medium (thread PRRT_kwDOJKSTjM58rtx1): the platform's
    ``Requests`` helper retries *indefinitely* on 429/5xx by default
    (no ``stop`` condition in tenacity unless ``retry_max_attempts`` is
    passed). If Google's revoke endpoint transiently returns a 429 or
    5xx, ``revoke_tokens`` would hang forever and block the credential
    deletion API call.

    Pin the bound: ``Requests`` must be constructed with a finite
    ``retry_max_attempts`` so revoke always terminates."""
    mock_response = MagicMock()
    mock_response.ok = True
    mock_post = AsyncMock(return_value=mock_response)
    mock_requests_cls = mocker.patch(
        "backend.integrations.oauth.google.Requests",
        return_value=MagicMock(post=mock_post),
    )

    await _handler().revoke_tokens(_creds())

    assert mock_requests_cls.call_count == 1
    _, kwargs = mock_requests_cls.call_args
    retry_max_attempts = kwargs.get("retry_max_attempts")
    assert retry_max_attempts is not None, (
        "Requests was constructed without retry_max_attempts — retries are "
        "unbounded, and a transient 429/5xx from Google would hang revoke."
    )
    assert retry_max_attempts >= 1
