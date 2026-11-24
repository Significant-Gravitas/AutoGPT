import asyncio
import logging
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import httpx
import jwt
import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr

from backend.api.features.integrations import sign_in_connection
from backend.api.rest_api import app as rest_app
from backend.data.model import OAuth2Credentials
from backend.integrations.oauth import google as google_oauth
from backend.integrations.oauth import microsoft_tokens
from backend.integrations.oauth.google import GoogleOAuthHandler
from backend.integrations.oauth.sign_in import SignInConnectionError
from backend.util import encryption

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
    """The user's saved connections, and what the endpoint writes. The account
    is set up (it has its personal org) unless a test says otherwise."""
    saved: list[OAuth2Credentials] = []
    manager = MagicMock()
    manager.store.get_creds_by_provider = AsyncMock(side_effect=lambda *_: saved)
    manager.create = AsyncMock()
    upgrade = AsyncMock()
    has_org = AsyncMock(return_value=True)
    monkeypatch.setattr(sign_in_connection, "creds_manager", manager)
    monkeypatch.setattr(sign_in_connection, "_upgrade_existing_credential", upgrade)
    monkeypatch.setattr(sign_in_connection, "has_personal_org", has_org)
    app.dependency_overrides[sign_in_connection.signed_in_user] = lambda: "user-1"
    yield MagicMock(
        saved=saved, create=manager.create, upgrade=upgrade, has_org=has_org
    )
    app.dependency_overrides.clear()


class _FakeRedis:
    def __init__(self):
        self.values: dict[str, str] = {}
        self.expiry: dict[str, int | None] = {}

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.values[key] = value
        self.expiry[key] = ex

    async def getdel(self, key: str) -> str | None:
        self.expiry.pop(key, None)
        return self.values.pop(key, None)


@pytest.fixture
def held(monkeypatch):
    """Redis, where a sign-up's connection waits for its account, sealed with
    the backend's encryption key."""
    redis = _FakeRedis()
    monkeypatch.setattr(
        sign_in_connection, "get_redis_async", AsyncMock(return_value=redis)
    )
    monkeypatch.setattr(encryption, "ENCRYPTION_KEY", Fernet.generate_key().decode())
    return redis


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


def test_a_google_check_that_cant_resolve_google_is_refused(store, google, monkeypatch):
    class DNSFails:
        def __init__(self, *args, **kwargs):
            pass

        async def post(self, url, *args, **kwargs):
            # What Requests raises before sending when DNS fails.
            raise ValueError("Unable to resolve IP address for hostname")

    monkeypatch.setattr(google_oauth, "Requests", DNSFails)

    res = client.post("/api/auth/sign-in-connection", json=GOOGLE_BODY)

    assert res.status_code == 422
    store.create.assert_not_called()


def test_a_google_check_that_answers_with_no_token_info_is_refused(
    store, google, monkeypatch
):
    class NotJSON:
        ok = True

        def json(self):
            raise ValueError("Expecting value: line 1 column 1 (char 0)")

    class AnswersAPage:
        def __init__(self, *args, **kwargs):
            pass

        async def post(self, url, *args, **kwargs):
            return NotJSON()

    monkeypatch.setattr(google_oauth, "Requests", AnswersAPage)

    res = client.post("/api/auth/sign-in-connection", json=GOOGLE_BODY)

    assert res.status_code == 422
    store.create.assert_not_called()


def test_a_google_sign_in_without_googles_handler_is_refused(store, monkeypatch):
    monkeypatch.setattr(
        sign_in_connection, "_get_provider_oauth_handler", lambda _: MagicMock()
    )

    res = client.post("/api/auth/sign-in-connection", json=GOOGLE_BODY)

    assert res.status_code == 501
    store.create.assert_not_called()


def test_a_sign_in_never_widens_a_managed_or_system_connection(
    store, google, monkeypatch
):
    # The same Google account, but one connection the platform manages and
    # one of its own keys: neither is the account's to widen.
    managed = _google_connection("pat@gmail.com", [GMAIL]).model_copy(
        update={"is_managed": True}
    )
    system = _google_connection("pat@gmail.com", [GMAIL])
    monkeypatch.setattr(
        sign_in_connection,
        "is_system_credential",
        lambda credential_id: credential_id == system.id,
    )
    store.saved.extend([managed, system])

    res = client.post("/api/auth/sign-in-connection", json=GOOGLE_BODY)

    assert res.json() == {"result": "created"}
    store.upgrade.assert_not_called()


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
    monkeypatch.setattr(
        sign_in_connection,
        "microsoft_app",
        lambda: MagicMock(client_id="connection-app"),
    )

    res = client.post("/api/auth/sign-in-connection", json=MICROSOFT_BODY)

    assert res.status_code == 409
    store.create.assert_not_called()


def test_a_microsoft_sign_in_whose_keys_cant_be_fetched_is_refused(store, monkeypatch):
    # Microsoft's signing keys out of reach: the token can't be checked.
    built = AsyncMock(side_effect=jwt.PyJWKClientError("Fail to fetch data"))
    monkeypatch.setattr(sign_in_connection, "credentials_from_tokens", built)
    monkeypatch.setattr(
        sign_in_connection,
        "microsoft_app",
        lambda: MagicMock(client_id="connection-app"),
    )

    res = client.post("/api/auth/sign-in-connection", json=MICROSOFT_BODY)

    assert res.status_code == 422
    store.create.assert_not_called()


def test_a_microsoft_sign_in_the_app_cant_renew_is_refused(store, monkeypatch):
    # The frontend's answer from a few minutes ago still names the app, but
    # the app has lost its secret since.
    _deployment(
        monkeypatch,
        microsoft_client_id="connection-app",
        microsoft_client_secret="",
    )
    built = AsyncMock(return_value=_microsoft_connection())
    monkeypatch.setattr(sign_in_connection, "credentials_from_tokens", built)

    res = client.post("/api/auth/sign-in-connection", json=MICROSOFT_BODY)

    assert res.status_code == 501
    built.assert_not_called()
    store.create.assert_not_called()


def _deployment(monkeypatch, **secrets: str) -> None:
    """This deployment's Google and Microsoft app settings."""

    def settings():
        return SimpleNamespace(secrets=SimpleNamespace(**secrets))

    monkeypatch.setattr(sign_in_connection, "Settings", settings)
    monkeypatch.setattr(microsoft_tokens, "Settings", settings)


def test_anyone_can_ask_which_clients_the_connections_use(monkeypatch):
    _deployment(
        monkeypatch,
        google_client_id="integration-client",
        google_client_secret="secret",  # pragma: allowlist secret
        microsoft_client_id="connection-app",
        microsoft_client_secret="secret",  # pragma: allowlist secret
    )

    # No token: the frontend's server asks before anyone has signed in.
    res = client.get("/api/auth/sign-in-connection")

    assert res.status_code == 200
    assert res.json() == {"google": "integration-client", "microsoft": "connection-app"}


@pytest.mark.parametrize(
    "secrets",
    [
        pytest.param(
            {
                "google_client_id": "integration-client",
                "google_client_secret": "",
                "microsoft_client_id": "connection-app",
                "microsoft_client_secret": "",
            },
            id="clients-without-their-secrets",
        ),
        # Without an app of its own the backend falls back to AutoGPT's public
        # Microsoft client.
        pytest.param(
            {
                "google_client_id": "",
                "google_client_secret": "",
                "microsoft_client_id": "",
                "microsoft_client_secret": "",
            },
            id="no-clients",
        ),
    ],
)
def test_connections_that_cant_renew_a_sign_in_name_no_client(monkeypatch, secrets):
    _deployment(monkeypatch, **secrets)

    res = client.get("/api/auth/sign-in-connection")

    assert res.json() == {"google": None, "microsoft": None}


def test_refusals_never_echo_a_token(store, google, caplog):
    google.answer["body"]["azp"] = google.answer["body"]["aud"] = "login-client"

    res = client.post("/api/auth/sign-in-connection", json=GOOGLE_BODY)

    assert res.status_code == 409
    for secret in ("sign-in-access", "sign-in-refresh"):
        assert secret not in res.text
        assert secret not in caplog.text


def test_a_sign_ups_connection_waits_for_the_account_to_be_set_up(store, google, held):
    # The handoff comes as the sign-up completes, before the request that
    # sets the account up (and gives it the org a connection is stored in).
    store.has_org.return_value = False

    res = client.post("/api/auth/sign-in-connection", json=GOOGLE_BODY)

    assert res.status_code == 200, res.text
    assert res.json() == {"result": "held"}
    store.create.assert_not_called()
    key = "sign-in-connection:user-1"
    assert held.expiry[key] == sign_in_connection.HELD_FOR_SECONDS
    # Encrypted while it waits.
    for secret in ("sign-in-access", "sign-in-refresh"):
        assert secret not in held.values[key]


def test_an_account_set_up_while_its_connection_was_held_keeps_it(store, google, held):
    # Set up after the handoff looked and before it held the connection: the
    # account's own look for a held one found nothing.
    store.has_org.side_effect = [False, True]

    res = client.post("/api/auth/sign-in-connection", json=GOOGLE_BODY)

    assert res.json() == {"result": "created"}
    store.create.assert_called_once()
    assert held.values == {}


def test_a_held_connection_kept_by_neither_request_isnt_reported_as_held(
    store, google, held
):
    # The account was set up meanwhile, but keeping the connection then failed:
    # it's gone, not waiting.
    store.has_org.side_effect = [False, True]
    store.create.side_effect = RuntimeError("database unavailable")

    res = client.post("/api/auth/sign-in-connection", json=GOOGLE_BODY)

    assert res.json() == {"result": "skipped"}
    assert held.values == {}


def test_a_connection_that_cant_be_held_never_fails_the_sign_up(
    store, google, monkeypatch, caplog
):
    store.has_org.return_value = False
    monkeypatch.setattr(
        sign_in_connection,
        "get_redis_async",
        AsyncMock(side_effect=ConnectionError("Redis is down")),
    )

    with caplog.at_level(logging.WARNING):
        res = client.post("/api/auth/sign-in-connection", json=GOOGLE_BODY)

    assert res.status_code == 200
    assert res.json() == {"result": "skipped"}
    assert "A sign-up's connection couldn't be held" in caplog.text
    for secret in ("sign-in-access", "sign-in-refresh"):
        assert secret not in caplog.text


# How long the frontend waits for a sign-in's handoff (sign-in-connection.ts).
_FRONTEND_HANDOFF_WAIT_SECONDS = 8


def _api() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


# The tests that await run on pytest's own loop: asyncio.run() would leave the
# thread without a current loop for the sync tests after them.
async def test_setting_the_account_up_keeps_its_held_connection_once(
    store, google, held
):
    store.has_org.return_value = False
    async with _api() as api:
        await api.post("/api/auth/sign-in-connection", json=GOOGLE_BODY)

    result = await sign_in_connection.keep_held_sign_in_connection("user-1")

    assert result == "created"
    (user_id, created), _ = store.create.call_args
    assert user_id == "user-1"
    assert created.username == "pat@gmail.com"
    assert set(created.scopes) == {"openid", GMAIL, CALENDAR}
    assert created.access_token.get_secret_value() == "ya29.sign-in-access"
    assert created.refresh_token.get_secret_value() == "1//sign-in-refresh"
    # Taken, so the account's next sign-in finds nothing waiting.
    assert held.values == {}
    assert await sign_in_connection.keep_held_sign_in_connection("user-1") is None
    assert store.create.call_count == 1


async def test_a_held_connection_that_cant_be_read_is_dropped_quietly(
    store, held, caplog
):
    held.values["sign-in-connection:user-1"] = "ya29.not-sealed"

    with caplog.at_level(logging.WARNING):
        result = await sign_in_connection.keep_held_sign_in_connection("user-1")

    # Taken and gone, so not still held.
    assert result == "skipped"
    store.create.assert_not_called()
    assert "A held sign-in connection couldn't be read" in caplog.text
    assert "ya29.not-sealed" not in caplog.text


# Many times the lookup's (patched) timeout, and nowhere near the hour a hang
# would take: a stalled CI runner can't fail it.
_HANG_SECONDS = 10


async def test_an_unreachable_redis_never_holds_up_a_sign_in(store, monkeypatch):
    async def never_connects():
        await asyncio.sleep(3600)

    monkeypatch.setattr(sign_in_connection, "get_redis_async", never_connects)
    monkeypatch.setattr(sign_in_connection, "HELD_LOOKUP_TIMEOUT_SECONDS", 0.05)
    started = time.monotonic()

    result = await sign_in_connection.keep_held_sign_in_connection("user-1")

    assert result is None
    assert time.monotonic() - started < _HANG_SECONDS


async def test_googles_token_check_ends_before_the_frontend_stops_waiting(
    monkeypatch,
):
    # The whole check, however many throttled answers Google retries through.
    assert google_oauth.TOKEN_INFO_TIMEOUT_SECONDS < _FRONTEND_HANDOFF_WAIT_SECONDS

    class GoogleNeverAnswers:
        def __init__(self, *args, **kwargs):
            pass

        async def post(self, *args, **kwargs):
            await asyncio.sleep(3600)

    monkeypatch.setattr(google_oauth, "Requests", GoogleNeverAnswers)
    monkeypatch.setattr(google_oauth, "TOKEN_INFO_TIMEOUT_SECONDS", 0.05)
    handler = GoogleOAuthHandler(
        client_id="integration-client",
        client_secret="secret",
        redirect_uri="https://platform.example/auth/integrations/oauth_callback",
    )
    started = time.monotonic()

    # A refusal like any other: the route answers 422, never a 500.
    with pytest.raises(SignInConnectionError):
        await handler.credentials_from_sign_in("ya29.sign-in-access", None, None)
    assert time.monotonic() - started < _HANG_SECONDS
