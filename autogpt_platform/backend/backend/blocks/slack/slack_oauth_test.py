"""Tests for the Slack OAuth handler."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks.slack._oauth import SlackOAuthError, SlackOAuthHandler
from backend.data.model import OAuth2Credentials


def _mock_response(json_data: dict[str, Any]) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = json_data
    return resp


def _make_handler(**overrides) -> SlackOAuthHandler:
    defaults = {
        "client_id": "test-client-id",
        "client_secret": "test-client-secret",
        "redirect_uri": "https://app.example.com/callback",
    }
    defaults.update(overrides)
    return SlackOAuthHandler(**defaults)


def _oauth_response(**overrides) -> dict[str, Any]:
    base: dict[str, Any] = {
        "ok": True,
        "access_token": "xoxb-new-bot-token",
        "token_type": "bot",
        "scope": "chat:write,chat:write.public",
        "bot_user_id": "U0KRQLJ9H",
        "app_id": "A0KRD7HC3",
        "team": {"id": "T9TK3CUKW", "name": "Test Workspace"},
        "authed_user": {"id": "U1234"},
    }
    base.update(overrides)
    return base


class TestGetLoginUrl:
    def test_basic(self):
        handler = _make_handler()
        url = handler.get_login_url(
            scopes=["chat:write", "chat:write.public"],
            state="random-state-token",
            code_challenge=None,
        )

        assert "https://slack.com/oauth/v2/authorize?" in url
        assert "client_id=test-client-id" in url
        assert "state=random-state-token" in url
        assert "scope=chat%3Awrite%2Cchat%3Awrite.public" in url

    def test_uses_default_scopes_when_empty(self):
        handler = _make_handler()
        url = handler.get_login_url(scopes=[], state="state", code_challenge=None)

        assert "scope=chat%3Awrite" in url

    def test_ignores_pkce_code_challenge(self):
        """Slack's OAuth v2 endpoint doesn't support PKCE — a code_challenge
        must never leak into the authorize URL."""
        handler = _make_handler()
        url = handler.get_login_url(
            scopes=["chat:write"], state="state", code_challenge="some-challenge"
        )

        assert "code_challenge" not in url


class TestExchangeCodeForTokens:
    @pytest.mark.asyncio
    async def test_success_returns_bot_token(self):
        handler = _make_handler()
        resp = _mock_response(_oauth_response())

        with patch("backend.blocks.slack._oauth.Requests") as MockRequests:
            MockRequests.return_value.post = AsyncMock(return_value=resp)
            creds = await handler.exchange_code_for_tokens(
                code="auth-code", scopes=["chat:write"], code_verifier=None
            )

        assert isinstance(creds, OAuth2Credentials)
        assert creds.access_token.get_secret_value() == "xoxb-new-bot-token"
        assert creds.scopes == ["chat:write", "chat:write.public"]
        assert creds.refresh_token is None
        assert creds.access_token_expires_at is None

    @pytest.mark.asyncio
    async def test_stores_team_and_bot_metadata(self):
        handler = _make_handler()
        resp = _mock_response(_oauth_response())

        with patch("backend.blocks.slack._oauth.Requests") as MockRequests:
            MockRequests.return_value.post = AsyncMock(return_value=resp)
            creds = await handler.exchange_code_for_tokens(
                code="auth-code", scopes=["chat:write"], code_verifier=None
            )

        assert creds.username == "Test Workspace"
        assert creds.metadata["team_id"] == "T9TK3CUKW"
        assert creds.metadata["team_name"] == "Test Workspace"
        assert creds.metadata["bot_user_id"] == "U0KRQLJ9H"
        assert creds.metadata["app_id"] == "A0KRD7HC3"

    @pytest.mark.asyncio
    async def test_rotating_token_carries_refresh_token_and_expiry(self):
        """When the installing app has token rotation enabled, Slack returns
        expires_in/refresh_token alongside the access token."""
        handler = _make_handler()
        resp = _mock_response(
            _oauth_response(
                access_token="xoxe.xoxb-rotating-token",
                expires_in=43200,
                refresh_token="xoxe-1-refresh-token",
            )
        )

        with patch("backend.blocks.slack._oauth.Requests") as MockRequests:
            MockRequests.return_value.post = AsyncMock(return_value=resp)
            creds = await handler.exchange_code_for_tokens(
                code="auth-code", scopes=["chat:write"], code_verifier=None
            )

        assert creds.refresh_token is not None
        assert creds.refresh_token.get_secret_value() == "xoxe-1-refresh-token"
        assert creds.access_token_expires_at is not None

    @pytest.mark.asyncio
    async def test_ok_false_on_http_200_raises(self):
        """Slack signals OAuth failure with ok: false even on an HTTP 200
        response — a bare status-code check would miss this entirely."""
        handler = _make_handler()
        resp = _mock_response({"ok": False, "error": "invalid_code"})

        with patch("backend.blocks.slack._oauth.Requests") as MockRequests:
            MockRequests.return_value.post = AsyncMock(return_value=resp)
            with pytest.raises(SlackOAuthError, match="invalid_code"):
                await handler.exchange_code_for_tokens(
                    code="bad-code", scopes=["chat:write"], code_verifier=None
                )

    @pytest.mark.asyncio
    async def test_ok_false_without_error_field_uses_fallback(self):
        handler = _make_handler()
        resp = _mock_response({"ok": False})

        with patch("backend.blocks.slack._oauth.Requests") as MockRequests:
            MockRequests.return_value.post = AsyncMock(return_value=resp)
            with pytest.raises(SlackOAuthError, match="unknown_error"):
                await handler.exchange_code_for_tokens(
                    code="bad-code", scopes=["chat:write"], code_verifier=None
                )


class TestRefreshTokens:
    @pytest.mark.asyncio
    async def test_no_refresh_token_is_a_noop(self):
        """Classic (non-rotating) Slack bot tokens never issue a refresh
        token, so refreshing must return the credentials unchanged rather
        than error."""
        handler = _make_handler()
        creds = OAuth2Credentials(
            id="existing-id",
            provider="slack",
            access_token=SecretStr("xoxb-classic-token"),
            refresh_token=None,
            scopes=["chat:write"],
            title="test",
        )

        refreshed = await handler._refresh_tokens(creds)

        assert refreshed is creds

    @pytest.mark.asyncio
    async def test_rotating_token_refresh_preserves_id(self):
        handler = _make_handler()
        existing_creds = OAuth2Credentials(
            id="existing-id",
            provider="slack",
            access_token=SecretStr("xoxe.xoxb-old-token"),
            refresh_token=SecretStr("xoxe-1-old-refresh"),
            scopes=["chat:write"],
            title="test",
            metadata={"team_id": "T9TK3CUKW"},
        )

        resp = _mock_response(
            _oauth_response(
                access_token="xoxe.xoxb-refreshed-token",
                expires_in=43200,
                refresh_token="xoxe-1-new-refresh",
            )
        )

        with patch("backend.blocks.slack._oauth.Requests") as MockRequests:
            MockRequests.return_value.post = AsyncMock(return_value=resp)
            refreshed = await handler._refresh_tokens(existing_creds)

        assert refreshed.id == "existing-id"
        assert refreshed.access_token.get_secret_value() == "xoxe.xoxb-refreshed-token"
        assert refreshed.refresh_token is not None
        assert refreshed.refresh_token.get_secret_value() == "xoxe-1-new-refresh"
        # Metadata from the original credentials must be preserved, not dropped.
        assert refreshed.metadata["team_id"] == "T9TK3CUKW"


class TestRevokeTokens:
    @pytest.mark.asyncio
    async def test_success(self):
        handler = _make_handler()
        creds = OAuth2Credentials(
            provider="slack",
            access_token=SecretStr("xoxb-token"),
            scopes=["chat:write"],
            title="test",
        )
        resp = _mock_response({"ok": True, "revoked": True})

        with patch("backend.blocks.slack._oauth.Requests") as MockRequests:
            MockRequests.return_value.post = AsyncMock(return_value=resp)
            result = await handler.revoke_tokens(creds)

        assert result is True

    @pytest.mark.asyncio
    async def test_failure_returns_false(self):
        handler = _make_handler()
        creds = OAuth2Credentials(
            provider="slack",
            access_token=SecretStr("xoxb-token"),
            scopes=["chat:write"],
            title="test",
        )
        resp = _mock_response({"ok": False, "error": "invalid_auth"})

        with patch("backend.blocks.slack._oauth.Requests") as MockRequests:
            MockRequests.return_value.post = AsyncMock(return_value=resp)
            result = await handler.revoke_tokens(creds)

        assert result is False
