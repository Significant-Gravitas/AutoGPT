"""A failed OAuth refresh says why, and a dead refresh token is not replayed.

The provider token endpoints are mocked at each handler's HTTP client, and a
4xx is raised with the platform's own ``http_status_error``: the exact
exception ``Requests(raise_for_status=True)`` raises in production.
"""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks.linear._oauth import LinearOAuthHandler
from backend.data.model import OAuth2Credentials
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.integrations.oauth.github import GitHubOAuthHandler
from backend.integrations.oauth.google import GoogleOAuthHandler
from backend.integrations.oauth.refresh_failure import (
    RECONNECT_REQUIRED_KEY,
    CredentialsNeedReconnectError,
    OAuthTokenRequestError,
    describe_refresh_failure,
    reconnect_required,
)
from backend.util.request import HTTPClientError, http_status_error

CLIENT = ("client-id", "client-secret", "https://localhost/callback")
ACCESS_TOKEN = "at-SECRET-access-0001"
REFRESH_TOKEN = "rt-SECRET-refresh-0001"


class FakeTokenEndpoint:
    """Stands in for a provider's token endpoint behind ``Requests``."""

    def __init__(self, status: int, payload: dict):
        self.status = status
        self.payload = payload
        self.calls: list[dict] = []

    async def post(self, url: str, data: dict | None = None, **kwargs) -> MagicMock:
        self.calls.append({"url": url, "data": data or {}})
        body = json.dumps(self.payload).encode()
        if self.status >= 400:
            raise http_status_error(self.status, "Bad Request", body)
        return MagicMock(
            ok=True,
            status=self.status,
            json=MagicMock(return_value=self.payload),
            text=MagicMock(return_value=body.decode()),
        )

    def patch(self, module: str):
        return patch(f"{module}.Requests", return_value=self)


def expiring(provider: str, scopes: list[str] | None = None) -> OAuth2Credentials:
    return OAuth2Credentials(
        id=f"{provider}-cred-1",
        provider=provider,
        title=f"My {provider}",
        username="alice",
        access_token=SecretStr(ACCESS_TOKEN),
        access_token_expires_at=1,  # long expired: needs_refresh() is True
        refresh_token=SecretStr(REFRESH_TOKEN),
        scopes=scopes or ["read"],
    )


@pytest.fixture(autouse=True)
def _no_redis_broadcast():
    with patch(
        "backend.integrations.creds_manager.publish_creds_changed",
        new_callable=AsyncMock,
    ):
        yield


@pytest.fixture
def sentry():
    with patch("backend.integrations.oauth.refresh_failure.sentry_sdk") as sdk:
        yield sdk


class InMemoryStore:
    """The one credential a test refreshes, persisted in memory."""

    def __init__(self, creds: OAuth2Credentials):
        self.creds = creds
        self.get_creds_by_id = AsyncMock(side_effect=lambda *_: self.creds)
        self.update_creds = AsyncMock(side_effect=self._update)

    async def _update(self, _user_id: str, updated: OAuth2Credentials) -> None:
        self.creds = updated


def manager_with(
    handler, stored: OAuth2Credentials
) -> tuple[IntegrationCredentialsManager, InMemoryStore]:
    manager = IntegrationCredentialsManager()
    store = InMemoryStore(stored)
    manager.store = MagicMock(
        get_creds_by_id=store.get_creds_by_id, update_creds=store.update_creds
    )
    manager._get_oauth_handler = AsyncMock(return_value=handler)
    return manager, store


def all_log_text(caplog) -> str:
    return "\n".join(
        f"{r.getMessage()} {getattr(r, 'json_fields', '')}" for r in caplog.records
    )


# -- 1. Nothing records why a refresh failed -- #


@pytest.mark.asyncio
async def test_linear_invalid_grant_is_logged_with_status_code_and_ids_but_no_token(
    caplog, sentry
):
    endpoint = FakeTokenEndpoint(
        400, {"error": "invalid_grant", "error_description": "Refresh token revoked"}
    )
    manager, store = manager_with(LinearOAuthHandler(*CLIENT), expiring("linear"))

    with endpoint.patch("backend.blocks.linear._oauth"), caplog.at_level(
        logging.WARNING
    ):
        with pytest.raises(CredentialsNeedReconnectError):
            await manager.refresh_if_needed("user-1", expiring("linear"), lock=False)

    report = next(r for r in caplog.records if "OAuth refresh failed" in r.message)
    assert report.levelno == logging.ERROR
    assert "linear credential #linear-cred-1" in report.getMessage()
    assert "HTTP 400" in report.getMessage()
    assert "error=invalid_grant" in report.getMessage()
    assert report.json_fields == {
        "provider": "linear",
        "credential_id": "linear-cred-1",
        "http_status": 400,
        "oauth_error": "invalid_grant",
        "definitive": True,
        "exception_type": "HTTPClientError",
    }
    logged = all_log_text(caplog)
    assert REFRESH_TOKEN not in logged and ACCESS_TOKEN not in logged


@pytest.mark.asyncio
async def test_refresh_failure_leaves_a_sentry_breadcrumb_and_tags(sentry):
    endpoint = FakeTokenEndpoint(400, {"error": "invalid_grant"})
    manager, store = manager_with(LinearOAuthHandler(*CLIENT), expiring("linear"))

    with endpoint.patch("backend.blocks.linear._oauth"):
        with pytest.raises(CredentialsNeedReconnectError):
            await manager.refresh_if_needed("user-1", expiring("linear"), lock=False)

    crumb = sentry.add_breadcrumb.call_args.kwargs
    assert crumb["category"] == "oauth.refresh"
    assert crumb["data"]["oauth_error"] == "invalid_grant"
    assert crumb["data"]["http_status"] == 400
    assert crumb["data"]["credential_id"] == "linear-cred-1"
    scope = sentry.new_scope.return_value.__enter__.return_value
    tags = {c.args[0]: c.args[1] for c in scope.set_tag.call_args_list}
    assert tags == {
        "oauth_refresh_provider": "linear",
        "oauth_refresh_status": "400",
        "oauth_refresh_error": "invalid_grant",
    }
    assert REFRESH_TOKEN not in repr(sentry.mock_calls)
    assert ACCESS_TOKEN not in repr(sentry.mock_calls)


@pytest.mark.asyncio
async def test_google_invalid_grant_reports_status_and_code(caplog, sentry):
    """AUTOGPT-SERVER-9FA: google-auth's RefreshError hides the status."""
    google_response = MagicMock(
        status=400,
        headers={},
        data=json.dumps(
            {"error": "invalid_grant", "error_description": "Bad Request"}
        ).encode(),
    )
    manager, store = manager_with(
        GoogleOAuthHandler(*CLIENT), expiring("google", ["openid"])
    )

    with patch(
        "google.auth.transport.requests.Request.__call__",
        return_value=google_response,
    ), caplog.at_level(logging.WARNING):
        with pytest.raises(CredentialsNeedReconnectError) as raised:
            await manager.refresh_if_needed(
                "user-1", expiring("google", ["openid"]), lock=False
            )

    assert isinstance(raised.value.__cause__, OAuthTokenRequestError)
    report = next(r for r in caplog.records if "OAuth refresh failed" in r.message)
    assert report.json_fields["http_status"] == 400
    assert report.json_fields["oauth_error"] == "invalid_grant"
    assert REFRESH_TOKEN not in all_log_text(caplog)


@pytest.mark.asyncio
async def test_github_error_in_a_200_body_is_recognised(sentry):
    endpoint = FakeTokenEndpoint(
        200,
        {
            "error": "bad_refresh_token",
            "error_description": "The refresh token passed is incorrect or expired.",
        },
    )
    manager, store = manager_with(
        GitHubOAuthHandler(*CLIENT), expiring("github", ["repo"])
    )

    with endpoint.patch("backend.integrations.oauth.github"):
        with pytest.raises(CredentialsNeedReconnectError) as raised:
            await manager.refresh_if_needed(
                "user-1", expiring("github", ["repo"]), lock=False
            )

    failure = describe_refresh_failure(raised.value.__cause__)
    assert (failure.status_code, failure.error_code) == (200, "bad_refresh_token")


@pytest.mark.asyncio
async def test_transient_failure_is_reported_but_not_marked(caplog, sentry):
    endpoint = FakeTokenEndpoint(400, {"error": "invalid_request"})
    manager, store = manager_with(LinearOAuthHandler(*CLIENT), expiring("linear"))

    with endpoint.patch("backend.blocks.linear._oauth"), caplog.at_level(
        logging.WARNING
    ):
        with pytest.raises(HTTPClientError) as raised:
            await manager.refresh_if_needed("user-1", expiring("linear"), lock=False)

    assert not isinstance(raised.value, CredentialsNeedReconnectError)
    report = next(r for r in caplog.records if "OAuth refresh failed" in r.message)
    assert report.levelno == logging.WARNING
    assert report.json_fields["oauth_error"] == "invalid_request"
    store.update_creds.assert_not_awaited()


# -- 2. Dead tokens get reused -- #


@pytest.mark.asyncio
async def test_invalid_grant_marks_the_credential_and_is_never_replayed(sentry):
    endpoint = FakeTokenEndpoint(400, {"error": "invalid_grant"})
    manager, store = manager_with(LinearOAuthHandler(*CLIENT), expiring("linear"))

    with endpoint.patch("backend.blocks.linear._oauth"):
        with pytest.raises(CredentialsNeedReconnectError):
            await manager.refresh_if_needed("user-1", expiring("linear"), lock=False)
        stored = store.creds
        marker = reconnect_required(stored)
        assert marker is not None
        assert (marker.error_code, marker.status_code) == ("invalid_grant", 400)
        assert REFRESH_TOKEN not in json.dumps(stored.metadata)

        # Later turns (unlocked copilot path and locked block path) read the
        # marked row and must not send the dead refresh token again.
        for lock in (False, True):
            manager._locked = MagicMock(return_value=_Noop())
            manager._acquire_lock = AsyncMock(return_value=_released_lock())
            with pytest.raises(CredentialsNeedReconnectError) as again:
                await manager.refresh_if_needed("user-1", stored, lock=lock)
            assert "reconnect" in str(again.value)
            assert "invalid_grant" in str(again.value)

    assert len(endpoint.calls) == 1


@pytest.mark.asyncio
async def test_a_successful_refresh_clears_the_marker(sentry):
    endpoint = FakeTokenEndpoint(
        200,
        {"access_token": "at-new", "refresh_token": "rt-new", "expires_in": 3600},
    )
    handler = LinearOAuthHandler(*CLIENT)
    creds = expiring("linear")
    creds.metadata = {"keep": "me"}
    manager, store = manager_with(handler, creds)

    with endpoint.patch("backend.blocks.linear._oauth"):
        fresh = await manager._refresh_and_store("user-1", creds, handler)

    assert RECONNECT_REQUIRED_KEY not in fresh.metadata
    assert reconnect_required(store.creds) is None


def test_the_reconnect_error_is_a_provider_refusal_callers_already_handle():
    """run_block turns an HTTPClientError into a reconnect card, not an error."""
    marked = expiring("linear")
    marked.metadata = {
        RECONNECT_REQUIRED_KEY: {"error_code": "invalid_grant", "status_code": 400}
    }
    marker = reconnect_required(marked)
    assert marker is not None
    error = CredentialsNeedReconnectError("linear", marked.id, marker)
    assert isinstance(error, HTTPClientError)
    assert error.status_code == 400
    assert str(error) == (
        "Linear refused to refresh the saved sign-in (invalid_grant, HTTP 400); "
        "reconnect the account to continue"
    )


@pytest.mark.parametrize(
    "exc,expected",
    [
        (
            http_status_error(400, "Bad Request", b'{"error": "invalid_grant"}'),
            (400, "invalid_grant"),
        ),
        (
            OAuthTokenRequestError("google", status_code=400, error_code="x_y"),
            (400, "x_y"),
        ),
        # A LinearAPIException-style message with no structured body.
        (
            RuntimeError("Failed to fetch Linear tokens (400): invalid_grant: nope"),
            (None, "invalid_grant"),
        ),
        (http_status_error(503, "Unavailable", b"<html>down</html>"), (503, None)),
        (ValueError("no refresh token"), (None, None)),
    ],
)
def test_describe_refresh_failure(exc, expected):
    failure = describe_refresh_failure(exc)
    assert (failure.status_code, failure.error_code) == expected


class _Noop:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *_):
        return None


def _released_lock() -> AsyncMock:
    lock = AsyncMock()
    lock.locked.return_value = False
    return lock
