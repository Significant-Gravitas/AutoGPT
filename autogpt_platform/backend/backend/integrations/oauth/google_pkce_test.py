"""PKCE regression tests for GoogleOAuthHandler.

The platform generates its own PKCE pair in ``store_state_token`` and hands
the challenge to ``get_login_url`` and the verifier to
``exchange_code_for_tokens``. The Google handler builds a fresh
``google_auth_oauthlib.flow.Flow`` for each step, so a verifier the library
generates for the login URL is thrown away before the exchange. Since
google-auth-oauthlib 1.3.0 ``Flow.from_client_config`` auto-generates that
verifier by default, which put a ``code_challenge`` on every Google login URL
with nothing to back it at the token endpoint. Google then rejects every
exchange with ``(invalid_grant) Missing code verifier.``

These tests pin the handler to the platform's PKCE pair on both legs.
"""

import time
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlparse

import pytest
from pytest_mock import MockerFixture

from backend.integrations.oauth.google import GoogleOAuthHandler

CODE_CHALLENGE = "platform-generated-challenge"
CODE_VERIFIER = "platform-generated-verifier"
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]


def _handler() -> GoogleOAuthHandler:
    return GoogleOAuthHandler(
        client_id="test-client-id",
        client_secret="test-client-secret",
        redirect_uri="https://example.com/auth/integrations/oauth_callback",
    )


def _login_params(handler: GoogleOAuthHandler, code_challenge: str | None):
    url = handler.get_login_url(SCOPES, "state-token", code_challenge=code_challenge)
    return parse_qs(urlparse(url).query)


def test_login_url_carries_the_platform_code_challenge():
    params = _login_params(_handler(), CODE_CHALLENGE)

    assert params["code_challenge"] == [CODE_CHALLENGE]
    assert params["code_challenge_method"] == ["S256"]
    assert params["state"] == ["state-token"]


def test_login_url_without_platform_challenge_has_no_pkce():
    """A caller that opts out of PKCE must not get a library-generated
    challenge, because nothing would send the matching verifier later."""
    params = _login_params(_handler(), None)

    assert "code_challenge" not in params
    assert "code_challenge_method" not in params


@pytest.mark.asyncio
async def test_exchange_sends_the_platform_code_verifier(mocker: MockerFixture):
    handler = _handler()
    captured: dict[str, object] = {}
    token = {
        "access_token": "access-token",
        "refresh_token": "refresh-token",
        "expires_in": 3600,
        "expires_at": time.time() + 3600,
        "scope": SCOPES,
        "token_type": "Bearer",
    }
    real_setup = handler._setup_oauth_flow

    def setup_with_stubbed_session(scopes):
        flow = real_setup(scopes)

        def fake_fetch_token(token_url, **kwargs):
            captured["token_url"] = token_url
            captured.update(kwargs)
            flow.oauth2session.token = token
            return token

        flow.oauth2session.fetch_token = MagicMock(side_effect=fake_fetch_token)
        return flow

    mocker.patch.object(handler, "_setup_oauth_flow", setup_with_stubbed_session)
    mocker.patch.object(handler, "_request_email", return_value="user@example.com")

    credentials = await handler.exchange_code_for_tokens(
        "auth-code", SCOPES, code_verifier=CODE_VERIFIER
    )

    assert captured["token_url"] == handler.token_uri
    assert captured["code"] == "auth-code"
    assert captured["code_verifier"] == CODE_VERIFIER
    assert credentials.username == "user@example.com"
    assert credentials.scopes == SCOPES
