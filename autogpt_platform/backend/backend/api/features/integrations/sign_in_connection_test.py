from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import jwt
import pytest
from pydantic import SecretStr

from backend.api.features.integrations import sign_in_connection
from backend.api.rest_api import app as rest_app
from backend.data.model import OAuth2Credentials
from backend.integrations.oauth import google as google_oauth
from backend.integrations.oauth.google import GoogleOAuthHandler

app = fastapi.FastAPI(exception_handlers=rest_app.exception_handlers)
# Mirror the prefix rest_api.py mounts it under.
app.include_router(sign_in_connection.sign_in_connection_router, prefix="/api/auth")
client = fastapi.testclient.TestClient(app, raise_server_exceptions=False)

GMAIL = "https://www.googleapis.com/auth/gmail.readonly"
CALENDAR = "https://www.googleapis.com/auth/calendar"
GOOGLE_BODY = {
    "provider": "google",
    "access_token": "ya29.sign-in-access",
    "refresh_token": "1//sign-in-refresh",
    "expires_at": 2_000_000_000,
    "scope": f"openid {GMAIL} {CALENDAR}",
}


def _google_connection(username: str, scopes: list[str]) -> OAuth2Credentials:
    return OAuth2Credentials(
        provider="google",
        title=None,
        username=username,
        access_token=SecretStr("old-access"),
        refresh_token=SecretStr("old-refresh"),
        access_token_expires_at=None,
        refresh_token_expires_at=None,
        scopes=scopes,
    )


class _TokenInfo:
    """Google's tokeninfo answer for the sign-in's access token."""

    def __init__(self, ok: bool, body: dict):
        self.ok = ok
        self._body = body

    def json(self) -> dict:
        return self._body


@pytest.fixture
def store(monkeypatch):
    """The user's saved connections, and what the endpoint writes."""
    saved: list[OAuth2Credentials] = []
    manager = MagicMock()
    manager.store.get_creds_by_provider = AsyncMock(side_effect=lambda *_: saved)
    manager.create = AsyncMock()
    upgrade = AsyncMock()
    monkeypatch.setattr(sign_in_connection, "creds_manager", manager)
    monkeypatch.setattr(sign_in_connection, "_upgrade_existing_credential", upgrade)
    app.dependency_overrides[sign_in_connection.signed_in_user] = lambda: "user-1"
    yield MagicMock(saved=saved, create=manager.create, upgrade=upgrade)
    app.dependency_overrides.clear()


@pytest.fixture
def google(monkeypatch):
    """The backend's Google client, and the tokeninfo Google answers with."""
    handler = GoogleOAuthHandler(
        client_id="integration-client",
        client_secret="secret",
        redirect_uri="https://platform.example/auth/integrations/oauth_callback",
    )
    monkeypatch.setattr(
        sign_in_connection, "_get_provider_oauth_handler", lambda _: handler
    )
    answer = {
        "ok": True,
        "body": {
            "azp": "integration-client",
            "aud": "integration-client",
            "email": "pat@gmail.com",
            "scope": f"openid {GMAIL} {CALENDAR}",
            "expires_in": "3599",
        },
    }
    posts: list[dict] = []

    class FakeRequests:
        def __init__(self, *args, **kwargs):
            pass

        async def post(self, url, *args, **kwargs):
            posts.append({"url": url, **kwargs})
            return _TokenInfo(answer["ok"], answer["body"])

    monkeypatch.setattr(google_oauth, "Requests", FakeRequests)
    return MagicMock(answer=answer, posts=posts)


def test_only_the_frontends_sign_in_token_is_accepted():
    res = client.post("/api/auth/sign-in-connection", json=GOOGLE_BODY)

    assert res.status_code == 401


def test_a_google_sign_in_starts_the_accounts_connection(store, google):
    res = client.post("/api/auth/sign-in-connection", json=GOOGLE_BODY)

    assert res.status_code == 200, res.text
    assert res.json() == {"result": "created"}
    (user_id, created), _ = store.create.call_args
    assert user_id == "user-1"
    assert created.provider == "google"
    assert created.username == "pat@gmail.com"
    assert set(created.scopes) == {"openid", GMAIL, CALENDAR}
    assert created.access_token.get_secret_value() == "ya29.sign-in-access"
    assert created.refresh_token.get_secret_value() == "1//sign-in-refresh"
    assert created.access_token_expires_at == 2_000_000_000
    # The token goes to Google in the body, never in a URL.
    assert google.posts[0]["url"] == GoogleOAuthHandler.TOKEN_INFO_ENDPOINT
    assert google.posts[0]["data"] == {"access_token": "ya29.sign-in-access"}


def test_a_token_from_another_google_client_is_refused(store, google):
    # Google renews a refresh token only for the client it was issued to.
    google.answer["body"]["azp"] = google.answer["body"]["aud"] = "login-client"

    res = client.post("/api/auth/sign-in-connection", json=GOOGLE_BODY)

    assert res.status_code == 409
    store.create.assert_not_called()
    store.upgrade.assert_not_called()


def test_a_token_google_refuses_stores_nothing(store, google):
    google.answer["ok"] = False

    res = client.post("/api/auth/sign-in-connection", json=GOOGLE_BODY)

    assert res.status_code == 422
    store.create.assert_not_called()


def test_a_sign_in_widens_the_accounts_connection_it_covers(store, google):
    store.saved.append(_google_connection("pat@gmail.com", [GMAIL]))

    res = client.post("/api/auth/sign-in-connection", json=GOOGLE_BODY)

    assert res.json() == {"result": "updated"}
    (user_id, saved_id, new), _ = store.upgrade.call_args
    assert (user_id, saved_id) == ("user-1", store.saved[0].id)
    assert set(new.scopes) == {"openid", GMAIL, CALENDAR}
    store.create.assert_not_called()


def test_a_sign_in_never_adds_a_second_connection_for_the_account(store, google):
    # Granted more than this sign-in asked for, e.g. from the Connect flow.
    wider = [GMAIL, CALENDAR, "https://www.googleapis.com/auth/drive.file"]
    store.saved.append(_google_connection("pat@gmail.com", wider))

    res = client.post("/api/auth/sign-in-connection", json=GOOGLE_BODY)

    assert res.json() == {"result": "kept"}
    store.create.assert_not_called()
    store.upgrade.assert_not_called()


def test_another_google_account_gets_its_own_connection(store, google):
    store.saved.append(_google_connection("pat@company.example", [GMAIL]))

    res = client.post("/api/auth/sign-in-connection", json=GOOGLE_BODY)

    assert res.json() == {"result": "created"}
    store.upgrade.assert_not_called()


def test_without_a_refresh_token_a_sign_in_starts_no_connection(store, google):
    # Google issues one only on a consent screen; an access token alone would
    # stop working within the hour.
    body = {**GOOGLE_BODY, "refresh_token": None}

    res = client.post("/api/auth/sign-in-connection", json=body)

    assert res.json() == {"result": "skipped"}
    store.create.assert_not_called()


def test_a_google_token_naming_no_account_starts_no_connection(store, google):
    del google.answer["body"]["email"]

    res = client.post("/api/auth/sign-in-connection", json=GOOGLE_BODY)

    assert res.json() == {"result": "skipped"}
    store.create.assert_not_called()


def _microsoft_connection() -> OAuth2Credentials:
    return OAuth2Credentials(
        provider="microsoft",
        title="Pat (personal account)",
        username="pat@outlook.com",
        access_token=SecretStr("eyJ.microsoft-access"),
        refresh_token=SecretStr("M.microsoft-refresh"),
        access_token_expires_at=None,
        refresh_token_expires_at=None,
        scopes=["Mail.ReadWrite", "Files.ReadWrite"],
        metadata={
            "microsoft_identity": {
                "tenant_id": "9188040d-6c67-4c5b-b112-36a304b66dad",
                "account_id": "oid-1",
                "account_type": "personal",
            }
        },
    )


MICROSOFT_BODY = {
    "provider": "microsoft",
    "access_token": "eyJ.microsoft-access",
    "refresh_token": "M.microsoft-refresh",
    "expires_at": None,
    "scope": "openid profile email offline_access Mail.ReadWrite Files.ReadWrite",
    "id_token": "eyJ.id-token",
}


def test_a_microsoft_sign_in_uses_the_connection_apps_own_checks(store, monkeypatch):
    built = AsyncMock(return_value=_microsoft_connection())
    monkeypatch.setattr(sign_in_connection, "credentials_from_tokens", built)
    monkeypatch.setattr(
        sign_in_connection,
        "microsoft_app",
        lambda: MagicMock(client_id="connection-app"),
    )

    res = client.post("/api/auth/sign-in-connection", json=MICROSOFT_BODY)

    assert res.json() == {"result": "created"}
    (payload,), kwargs = built.call_args
    # The ID token is checked against the backend's own app, the only client
    # that can renew the refresh token.
    assert kwargs == {"client_id": "connection-app", "auth_flow": "authorization_code"}
    assert payload["id_token"] == "eyJ.id-token"
    assert payload["refresh_token"] == "M.microsoft-refresh"
    assert payload["scope"] == MICROSOFT_BODY["scope"]


def test_a_microsoft_sign_in_from_another_app_is_refused(store, monkeypatch):
    built = AsyncMock(side_effect=jwt.InvalidAudienceError("aud"))
    monkeypatch.setattr(sign_in_connection, "credentials_from_tokens", built)

    res = client.post("/api/auth/sign-in-connection", json=MICROSOFT_BODY)

    assert res.status_code == 409
    store.create.assert_not_called()


def test_refusals_never_echo_a_token(store, google, caplog):
    google.answer["body"]["azp"] = google.answer["body"]["aud"] = "login-client"

    res = client.post("/api/auth/sign-in-connection", json=GOOGLE_BODY)

    assert res.status_code == 409
    for secret in ("sign-in-access", "sign-in-refresh"):
        assert secret not in res.text
        assert secret not in caplog.text
