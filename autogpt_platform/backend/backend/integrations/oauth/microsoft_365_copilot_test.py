"""Entra's defaults are wrong for this integration in two specific ways.

Both are silent. `/common` lets someone sign in with an account that can
never hold a Copilot licence, and the failure lands after they have already
consented. Omitting `offline_access` gets no refresh token, and the
connection stops working about an hour later with nothing to explain it.

The third is refresh-token rotation: Entra issues a new refresh token on
every use, and a response that omits one means "keep the old one" rather
than "you have none". Dropping it there ends the connection at the first
refresh.
"""

from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from backend.integrations.oauth.microsoft_365_copilot import (
    Microsoft365CopilotOAuthHandler,
)


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeRequests:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.sent: dict[str, Any] = {}

    async def post(self, url: str, *args, **kwargs) -> FakeResponse:
        self.sent = {"url": url, **kwargs}
        return FakeResponse(self._payload)


def handler() -> Microsoft365CopilotOAuthHandler:
    return Microsoft365CopilotOAuthHandler(
        "client-id", "client-secret", "https://app.invalid/callback"
    )


def login_params(url: str) -> dict[str, list[str]]:
    return parse_qs(urlparse(url).query)


class TestTheLoginUrl:
    def test_uses_the_organizations_tenant_not_common(self) -> None:
        """The Chat API is delegated-only and needs a Copilot licence, which
        a personal Microsoft account cannot hold. `/common` would accept one
        and fail after consent."""
        url = handler().get_login_url([], "state", None)

        assert "/organizations/oauth2/v2.0/authorize" in url
        assert "/common/" not in url

    def test_asks_for_offline_access(self) -> None:
        """Without it Entra returns no refresh token and the connection dies
        quietly about an hour in."""
        params = login_params(handler().get_login_url([], "state", None))

        assert "offline_access" in params["scope"][0].split()

    def test_carries_pkce_when_given_a_challenge(self) -> None:
        params = login_params(handler().get_login_url([], "state", "a-challenge"))

        assert params["code_challenge"] == ["a-challenge"]
        assert params["code_challenge_method"] == ["S256"]

    def test_omits_pkce_entirely_when_there_is_none(self) -> None:
        """Sending an empty challenge is not the same as sending none --
        Entra rejects the former."""
        params = login_params(handler().get_login_url([], "state", None))

        assert "code_challenge" not in params
        assert "code_challenge_method" not in params

    def test_round_trips_the_state(self) -> None:
        params = login_params(handler().get_login_url([], "state-xyz", None))

        assert params["state"] == ["state-xyz"]


class TestExchangingTheCode:
    @pytest.mark.asyncio
    async def test_sends_the_verifier_and_keeps_what_comes_back(self) -> None:
        http = FakeRequests(
            {
                "access_token": "at",
                "refresh_token": "rt",
                "expires_in": 3600,
                "scope": "Mail.Read offline_access",
            }
        )
        subject = handler()
        subject_requests = http

        import backend.integrations.oauth.microsoft_365_copilot as module

        original = module.Requests
        module.Requests = lambda: subject_requests  # type: ignore[assignment]
        try:
            creds = await subject.exchange_code_for_tokens(
                "the-code", [], "the-verifier"
            )
        finally:
            module.Requests = original

        assert http.sent["data"]["code_verifier"] == "the-verifier"
        assert http.sent["data"]["grant_type"] == "authorization_code"
        assert creds.access_token.get_secret_value() == "at"
        assert creds.scopes == ["Mail.Read", "offline_access"]
        assert creds.access_token_expires_at is not None

    @pytest.mark.asyncio
    async def test_a_response_without_a_token_raises_with_the_reason(self) -> None:
        """Entra puts the actual problem in `error_description`, and it is
        usually something the user can act on -- a missing licence, an
        unconsented scope."""
        http = FakeRequests(
            {"error": "invalid_grant", "error_description": "AADSTS70008: expired"}
        )

        import backend.integrations.oauth.microsoft_365_copilot as module

        original = module.Requests
        module.Requests = lambda: http  # type: ignore[assignment]
        try:
            with pytest.raises(ValueError) as raised:
                await handler().exchange_code_for_tokens("code", [], None)
        finally:
            module.Requests = original

        assert "AADSTS70008" in str(raised.value)


class TestRefreshing:
    @pytest.mark.asyncio
    async def test_keeps_the_old_refresh_token_when_none_comes_back(self) -> None:
        """Entra rotates on every use, and an omitted one means "keep what
        you have". Dropping it ends the connection at the first refresh."""
        from backend.data.model import OAuth2Credentials

        existing = OAuth2Credentials(
            provider="microsoft_365_copilot",
            access_token="old-at",
            refresh_token="old-rt",
            scopes=["Mail.Read"],
            title=None,
        )
        http = FakeRequests({"access_token": "new-at", "expires_in": 3600})

        import backend.integrations.oauth.microsoft_365_copilot as module

        original = module.Requests
        module.Requests = lambda: http  # type: ignore[assignment]
        try:
            refreshed = await handler()._refresh_tokens(existing)
        finally:
            module.Requests = original

        assert refreshed.access_token.get_secret_value() == "new-at"
        assert refreshed.refresh_token is not None
        assert refreshed.refresh_token.get_secret_value() == "old-rt"
        # The credential row is updated, not replaced -- a new id would
        # orphan every session already routed to this connection.
        assert refreshed.id == existing.id

    @pytest.mark.asyncio
    async def test_a_credential_with_no_refresh_token_is_left_alone(self) -> None:
        from backend.data.model import OAuth2Credentials

        existing = OAuth2Credentials(
            provider="microsoft_365_copilot",
            access_token="at",
            refresh_token=None,
            scopes=[],
            title=None,
        )

        assert await handler()._refresh_tokens(existing) is existing


class TestRevoking:
    @pytest.mark.asyncio
    async def test_does_not_claim_to_have_revoked_at_the_source(self) -> None:
        """Entra has no revocation endpoint for a delegated grant. The caller
        uses this to decide what to tell the user, and answering True would
        be a claim we cannot make."""
        from backend.data.model import OAuth2Credentials

        creds = OAuth2Credentials(
            provider="microsoft_365_copilot",
            access_token="at",
            scopes=[],
            title=None,
        )

        assert await handler().revoke_tokens(creds) is False
