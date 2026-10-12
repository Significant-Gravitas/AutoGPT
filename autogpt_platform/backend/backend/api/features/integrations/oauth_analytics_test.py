"""The OAuth login and callback report started / failed to PostHog."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import fastapi
import fastapi.testclient
import pytest
from fastapi import HTTPException
from pydantic import SecretStr

from backend.api.features.integrations.router import router
from backend.data.model import OAuth2Credentials, OAuthState
from backend.util import posthog_client

app = fastapi.FastAPI()
app.include_router(router)
client = fastapi.testclient.TestClient(app, raise_server_exceptions=False)

ROUTER = "backend.api.features.integrations.router"
AUTH_CODE = "4/0AVGzR1Bexample-auth-code"
STATE_TOKEN = "state-token-abc"


@pytest.fixture(autouse=True)
def setup_auth(mock_jwt_user):
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch) -> Mock:
    posthog = Mock()
    monkeypatch.setattr(posthog_client, "get_posthog_client", lambda: posthog)
    return posthog.capture


def _events(capture: Mock, name: str) -> list[dict]:
    return [
        c.kwargs["properties"]
        for c in capture.call_args_list
        if c.kwargs["event"] == name
    ]


def _only_failure(capture: Mock) -> dict:
    failures = _events(capture, "credential_oauth_exchange_failed")
    assert len(failures) == 1, f"expected 1 failure event, got {len(failures)}"
    return failures[0]


def _state(credential_id: str | None = None) -> OAuthState:
    return OAuthState(
        token=STATE_TOKEN,
        provider="google",
        expires_at=9999999999,
        scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        credential_id=credential_id,
    )


def _google_cred(cred_id: str = "google-cred-1", username: str = "alice@gmail.com"):
    return OAuth2Credentials(
        id=cred_id,
        provider="google",
        title="My Google",
        access_token=SecretStr("ya29.access-token"),
        refresh_token=SecretStr("1//refresh-token"),
        scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        username=username,
        access_token_expires_at=9999999999,
    )


def _handler(exchange: AsyncMock) -> MagicMock:
    handler = MagicMock()
    handler.handle_default_scopes.side_effect = lambda scopes: scopes
    handler.exchange_code_for_tokens = exchange
    return handler


def _post_callback(provider: str = "google"):
    return client.post(
        f"/{provider}/callback",
        json={"code": AUTH_CODE, "state_token": STATE_TOKEN},
    )


class TestOAuthStarted:
    def test_login_reports_the_provider(self, capture: Mock):
        handler = MagicMock()
        handler.get_login_url.return_value = "https://accounts.google.com/auth"
        with (
            patch(f"{ROUTER}._get_provider_oauth_handler", return_value=handler),
            patch(f"{ROUTER}.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.store_state_token = AsyncMock(
                return_value=(STATE_TOKEN, "code-challenge")
            )
            resp = client.get("/google/login", params={"scopes": "openid"})

        assert resp.status_code == 200
        started = _events(capture, "credential_oauth_started")
        assert len(started) == 1
        assert started[0]["provider"] == "google"
        assert started[0]["source"] == "platform"

    def test_login_that_fails_reports_nothing(self, capture: Mock):
        with patch(
            f"{ROUTER}._get_provider_oauth_handler",
            side_effect=HTTPException(status_code=501, detail="not configured"),
        ):
            resp = client.get("/google/login")

        assert resp.status_code == 501
        capture.assert_not_called()


class TestOAuthExchangeFailed:
    def test_invalid_state(self, capture: Mock):
        with patch(f"{ROUTER}.creds_manager") as mock_mgr:
            mock_mgr.store.verify_state_token = AsyncMock(return_value=None)
            resp = _post_callback()

        assert resp.status_code == 400
        failure = _only_failure(capture)
        assert failure["provider"] == "google"
        assert failure["failure_class"] == "invalid_state"
        assert failure["status_code"] == 400
        assert failure["detail"] == "Invalid or expired state token"

    def test_provider_unavailable(self, capture: Mock):
        unconfigured = HTTPException(
            status_code=501,
            detail={
                "message": "Integration with provider 'google' is not configured.",
                "hint": "Set client ID and secret",
            },
        )
        with (
            patch(f"{ROUTER}._get_provider_oauth_handler", side_effect=unconfigured),
            patch(f"{ROUTER}.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=_state())
            resp = _post_callback()

        assert resp.status_code == 501
        failure = _only_failure(capture)
        assert failure["failure_class"] == "provider_unavailable"
        assert failure["status_code"] == 501
        assert failure["detail"] == (
            "Integration with provider 'google' is not configured."
        )

    def test_token_exchange_reports_the_provider_error(self, capture: Mock):
        exchange = AsyncMock(
            side_effect=ValueError("(invalid_grant) Missing code verifier.")
        )
        with (
            patch(
                f"{ROUTER}._get_provider_oauth_handler",
                return_value=_handler(exchange),
            ),
            patch(f"{ROUTER}.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=_state())
            resp = _post_callback()

        assert resp.status_code == 400
        failure = _only_failure(capture)
        assert failure["provider"] == "google"
        assert failure["failure_class"] == "token_exchange"
        assert failure["status_code"] == 400
        assert failure["detail"] == "ValueError: (invalid_grant) Missing code verifier."

    def test_token_exchange_detail_hides_client_secret_and_code_verifier(
        self, capture: Mock
    ):
        exchange = AsyncMock(
            side_effect=ValueError(
                "rejected client_secret=shh verifier=pkce-v with invalid_grant"
            )
        )
        handler = _handler(exchange)
        handler.client_secret = "shh"
        state = _state()
        state.code_verifier = "pkce-v"
        with (
            patch(f"{ROUTER}._get_provider_oauth_handler", return_value=handler),
            patch(f"{ROUTER}.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=state)
            resp = _post_callback()

        assert resp.status_code == 400
        assert _only_failure(capture)["detail"] == (
            "ValueError: rejected client_secret=[redacted] "
            "verifier=[redacted] with invalid_grant"
        )

    def test_token_exchange_detail_is_truncated_and_carries_no_secrets(
        self, capture: Mock
    ):
        leaky = (
            f"code {AUTH_CODE} for alice@gmail.com rejected at "
            "https://oauth2.googleapis.com/token?code=abc&client_secret=shh "
            "with ya29.a0AfH6SMBx3example9token " + "x" * 400
        )
        exchange = AsyncMock(side_effect=ValueError(leaky))
        with (
            patch(
                f"{ROUTER}._get_provider_oauth_handler",
                return_value=_handler(exchange),
            ),
            patch(f"{ROUTER}.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=_state())
            resp = _post_callback()

        assert resp.status_code == 400
        detail = _only_failure(capture)["detail"]
        assert len(detail) <= 200
        assert detail.startswith("ValueError: code [redacted] for [email] rejected")
        for secret in (
            AUTH_CODE,
            "alice@gmail.com",
            "client_secret",
            "shh",
            "a0AfH6SMBx3example9token",
        ):
            assert secret not in detail

    def test_credential_merge(self, capture: Mock):
        existing = _google_cred(username="alice@gmail.com")
        new_cred = _google_cred(username="bob@gmail.com")
        with (
            patch(
                f"{ROUTER}._get_provider_oauth_handler",
                return_value=_handler(AsyncMock(return_value=new_cred)),
            ),
            patch(f"{ROUTER}.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(
                return_value=_state(credential_id="google-cred-1")
            )
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=existing)
            resp = _post_callback()

        assert resp.status_code == 400
        failure = _only_failure(capture)
        assert failure["failure_class"] == "credential_merge"
        assert failure["status_code"] == 400
        assert failure["detail"] == (
            "Username mismatch: authenticated as a different user"
        )

    def test_unexpected_error_reports_only_its_class(self, capture: Mock):
        with (
            patch(
                f"{ROUTER}._get_provider_oauth_handler",
                return_value=_handler(AsyncMock(return_value=_google_cred())),
            ),
            patch(f"{ROUTER}.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=_state())
            mock_mgr.store.get_creds_by_provider = AsyncMock(return_value=[])
            mock_mgr.create = AsyncMock(
                side_effect=RuntimeError("db down for alice@gmail.com")
            )
            resp = _post_callback()

        assert resp.status_code == 500
        failure = _only_failure(capture)
        # The step that raised stays the class. The status comes from the app's
        # exception handlers, which this router does not know, so it is left out.
        assert failure["failure_class"] == "credential_merge"
        assert "status_code" not in failure
        assert failure["detail"] == "RuntimeError"

    def test_success_reports_connected_and_no_failure(self, capture: Mock):
        with (
            patch(
                f"{ROUTER}._get_provider_oauth_handler",
                return_value=_handler(AsyncMock(return_value=_google_cred())),
            ),
            patch(f"{ROUTER}.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=_state())
            mock_mgr.store.get_creds_by_provider = AsyncMock(return_value=[])
            mock_mgr.create = AsyncMock()
            resp = _post_callback()

        assert resp.status_code == 200
        assert _events(capture, "credential_oauth_exchange_failed") == []
        connected = _events(capture, "integration_connected")
        assert len(connected) == 1
        assert connected[0]["provider"] == "google"
        assert connected[0]["method"] == "oauth"

    def test_codex_callback_is_not_reported(self, capture: Mock):
        with (
            patch(f"{ROUTER}.enforce_codex_access_http", AsyncMock()),
            patch(f"{ROUTER}.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=None)
            resp = _post_callback("codex")

        assert resp.status_code == 400
        assert _events(capture, "credential_oauth_exchange_failed") == []
