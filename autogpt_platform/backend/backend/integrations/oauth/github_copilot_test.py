"""Two GitHub behaviours here fail silently, and both are pinned below.

A classic PAT (``ghp_``) is accepted by GitHub, reaches the Copilot runtime,
and is then ignored with no error -- so a user who supplies one gets a
connection that links successfully and fails on every chat with nothing to
explain it.

And GitHub answers a *bad* token exchange with HTTP 200 and an error body.
A caller that checks only the status stores a credential with no token in
it, and the failure surfaces later somewhere unrelated.
"""

from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

import backend.integrations.oauth.github_copilot as module
from backend.data.model import OAuth2Credentials
from backend.integrations.oauth.github_copilot import GitHubCopilotOAuthHandler


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeRequests:
    def __init__(self, token_payload: dict[str, Any]) -> None:
        self._token_payload = token_payload
        self.sent: dict[str, Any] = {}

    async def post(self, url: str, *args, **kwargs) -> FakeResponse:
        self.sent = {"url": url, **kwargs}
        return FakeResponse(self._token_payload)

    async def get(self, url: str, *args, **kwargs) -> FakeResponse:
        return FakeResponse({"login": "octocat"})

    async def delete(self, *args, **kwargs) -> FakeResponse:
        return FakeResponse({})


def handler() -> GitHubCopilotOAuthHandler:
    return GitHubCopilotOAuthHandler(
        "client-id", "client-secret", "https://app.invalid/callback"
    )


async def with_http(payload: dict[str, Any], call):
    http = FakeRequests(payload)
    original = module.Requests
    module.Requests = lambda: http  # type: ignore[assignment]
    try:
        return await call(), http
    finally:
        module.Requests = original


class TestTheLoginUrl:
    def test_asks_for_read_user_and_nothing_copilot_shaped(self) -> None:
        """There is no `copilot` scope -- a seat grants access, not a scope.
        Asking for one that does not exist is rejected by GitHub outright."""
        url = handler().get_login_url([], "state", None)
        scopes = parse_qs(urlparse(url).query)["scope"][0].split()

        assert scopes == ["read:user"]
        assert not any("copilot" in scope for scope in scopes)

    def test_round_trips_the_state(self) -> None:
        params = parse_qs(urlparse(handler().get_login_url([], "s-1", None)).query)

        assert params["state"] == ["s-1"]


class TestExchangingTheCode:
    @pytest.mark.asyncio
    async def test_keeps_the_refresh_token_and_both_expiries(self) -> None:
        """A GitHub App user token lasts 8 hours with a 6-month refresh.
        Losing either expiry means we cannot tell a token that needs
        refreshing from one that needs re-linking."""
        creds, _ = await with_http(
            {
                "access_token": "ghu_abc",
                "refresh_token": "ghr_abc",
                "expires_in": 28800,
                "refresh_token_expires_in": 15897600,
                "scope": "read:user",
            },
            lambda: handler().exchange_code_for_tokens("code", [], None),
        )

        assert creds.access_token.get_secret_value() == "ghu_abc"
        assert creds.refresh_token is not None
        assert creds.access_token_expires_at is not None
        assert creds.refresh_token_expires_at is not None
        assert creds.username == "octocat"

    @pytest.mark.asyncio
    async def test_an_oauth_app_token_with_no_expiry_is_still_valid(self) -> None:
        """`gho_` tokens have no documented expiry and no refresh token.
        Treating the absence as an error would reject a working credential."""
        creds, _ = await with_http(
            {"access_token": "gho_abc", "scope": "read:user"},
            lambda: handler().exchange_code_for_tokens("code", [], None),
        )

        assert creds.access_token.get_secret_value() == "gho_abc"
        assert creds.refresh_token is None
        assert creds.access_token_expires_at is None

    @pytest.mark.asyncio
    async def test_a_classic_pat_is_refused_at_sign_in(self) -> None:
        """`ghp_` is accepted by GitHub and then ignored by the Copilot
        runtime with no error. Refusing here puts the failure where there is
        a reason to give, instead of on every chat with none."""
        with pytest.raises(ValueError) as raised:
            await with_http(
                {"access_token": "ghp_classic"},
                lambda: handler().exchange_code_for_tokens("code", [], None),
            )

        assert "classic personal access token" in str(raised.value)

    @pytest.mark.asyncio
    async def test_an_error_body_behind_a_200_is_still_an_error(self) -> None:
        """GitHub answers a bad exchange with 200 and an error body. A caller
        that trusts the status stores a credential with no token in it."""
        with pytest.raises(ValueError) as raised:
            await with_http(
                {
                    "error": "bad_verification_code",
                    "error_description": "The code passed is incorrect or expired.",
                },
                lambda: handler().exchange_code_for_tokens("code", [], None),
            )

        assert "incorrect or expired" in str(raised.value)


class TestRefreshing:
    @pytest.mark.asyncio
    async def test_a_credential_with_no_refresh_token_is_left_alone(self) -> None:
        """Not a failure: an OAuth App token, or a GitHub App with expiry
        turned off, has nothing to refresh and does not expire."""
        existing = OAuth2Credentials(
            provider="github_copilot",
            access_token="gho_abc",
            refresh_token=None,
            scopes=["read:user"],
            title=None,
        )

        assert await handler()._refresh_tokens(existing) is existing

    @pytest.mark.asyncio
    async def test_a_refresh_updates_the_row_rather_than_replacing_it(self) -> None:
        """A new id would orphan every session already routed here."""
        existing = OAuth2Credentials(
            provider="github_copilot",
            access_token="ghu_old",
            refresh_token="ghr_old",
            scopes=["read:user"],
            username="octocat",
            title=None,
        )

        refreshed, _ = await with_http(
            {
                "access_token": "ghu_new",
                "refresh_token": "ghr_new",
                "expires_in": 28800,
            },
            lambda: handler()._refresh_tokens(existing),
        )

        assert refreshed.access_token.get_secret_value() == "ghu_new"
        assert refreshed.id == existing.id
        # Not re-fetched: the account cannot change under a refresh, and a
        # failing display lookup must not break a working refresh.
        assert refreshed.username == "octocat"
