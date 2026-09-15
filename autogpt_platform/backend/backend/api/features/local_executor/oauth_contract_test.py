import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from prisma.enums import APIKeyPermission
from pydantic import SecretStr
from starlette.requests import Request

from backend.api.features import oauth
from backend.copilot.tools.local_pc_relay_protocol import RelayBackend
from backend.copilot.tools.local_pc_shim import ShimConnectionManager
from backend.data.auth.oauth import (
    InvalidClientError,
    OAuthApplicationInfoWithSecret,
    RefreshTokenFamilyRevokedError,
    validate_client_credentials,
)


def _make_client(*, raise_server_exceptions: bool = True) -> TestClient:
    app = FastAPI()
    app.include_router(oauth.router, prefix="/api/oauth")
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


def _issued_token(value: str):
    return SimpleNamespace(
        token=SecretStr(value),
        expires_at=datetime.now(timezone.utc),
        user_id="user-1",
        scopes=[APIKeyPermission.USE_TOOLS],
    )


_PKCE_VERIFIER = "a" * 43


class _NoopRelay:
    def __init__(self) -> None:
        self.revocations = 0

    async def revoke_owner(
        self, user_id: str, client_id: str | None, *, reason: str
    ) -> int:
        self.revocations += 1
        return 0


class _RetryingDirectWebSocket:
    def __init__(self) -> None:
        self.send_attempts = 0
        self.close_attempts = 0

    async def send_text(self, message: str) -> None:
        self.send_attempts += 1
        if self.send_attempts == 1:
            raise ConnectionError("direct send failed")

    async def iter_text(self):
        if False:
            yield ""

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.close_attempts += 1
        if self.close_attempts == 1:
            raise ConnectionError("direct close failed")


def _oauth_application(*, is_public: bool) -> OAuthApplicationInfoWithSecret:
    now = datetime.now(timezone.utc)
    return OAuthApplicationInfoWithSecret(
        id="app-1",
        name="Local Executor",
        client_id="autogpt-local-executor",
        redirect_uris=["http://localhost:41899/callback"],
        grant_types=["authorization_code", "refresh_token"],
        scopes=[APIKeyPermission.USE_TOOLS],
        owner_id="owner-1",
        is_active=True,
        is_public=is_public,
        created_at=now,
        updated_at=now,
        client_secret_hash="unused-hash",
        client_secret_salt="unused-salt",
    )


@pytest.mark.asyncio
async def test_public_client_credential_rules_are_grant_scoped() -> None:
    application = _oauth_application(is_public=True)
    with patch(
        "backend.data.auth.oauth.get_oauth_application_with_secret",
        AsyncMock(return_value=application),
    ):
        with pytest.raises(InvalidClientError, match="code_verifier"):
            await validate_client_credentials("autogpt-local-executor", "")
        with pytest.raises(InvalidClientError, match="RFC 7636"):
            await validate_client_credentials(
                "autogpt-local-executor", "", code_verifier="too-short"
            )

        by_pkce = await validate_client_credentials(
            "autogpt-local-executor", "", code_verifier=_PKCE_VERIFIER
        )
        by_refresh = await validate_client_credentials(
            "autogpt-local-executor", "", allow_public_without_secret=True
        )

    assert by_pkce.is_public is True
    assert by_refresh.is_public is True


@pytest.mark.asyncio
async def test_confidential_client_cannot_use_public_refresh_exception() -> None:
    application = _oauth_application(is_public=False)
    with (
        patch(
            "backend.data.auth.oauth.get_oauth_application_with_secret",
            AsyncMock(return_value=application),
        ),
        patch.object(
            OAuthApplicationInfoWithSecret, "verify_secret", return_value=False
        ),
    ):
        with pytest.raises(InvalidClientError, match="client_secret"):
            await validate_client_credentials(
                "autogpt-local-executor",
                "",
                allow_public_without_secret=True,
            )


@pytest.mark.parametrize("encoding", ["form", "json"])
def test_public_pkce_token_exchange_accepts_form_and_json(encoding: str) -> None:
    client = _make_client()
    application = SimpleNamespace(id="app-1", name="Local Executor")
    validate_client = AsyncMock(return_value=application)
    consume_code = AsyncMock(return_value=("user-1", [APIKeyPermission.USE_TOOLS]))
    payload = {
        "grant_type": "authorization_code",
        "code": "authorization-code",
        "redirect_uri": "http://localhost:41899/callback",
        "client_id": "autogpt-local-executor",
        "code_verifier": _PKCE_VERIFIER,
    }

    with (
        patch(
            "backend.api.features.oauth.validate_client_credentials",
            validate_client,
        ),
        patch("backend.api.features.oauth.consume_authorization_code", consume_code),
        patch(
            "backend.api.features.oauth.create_access_token",
            AsyncMock(return_value=_issued_token("access-token")),
        ),
        patch(
            "backend.api.features.oauth.create_refresh_token",
            AsyncMock(return_value=_issued_token("refresh-token")),
        ),
    ):
        if encoding == "form":
            response = client.post("/api/oauth/token", data=payload)
        else:
            response = client.post("/api/oauth/token", json=payload)

    assert response.status_code == 200
    assert response.json()["access_token"] == "access-token"
    validate_client.assert_awaited_once_with(
        "autogpt-local-executor", "", code_verifier=_PKCE_VERIFIER
    )


def test_public_refresh_form_uses_refresh_token_as_bearer_credential() -> None:
    client = _make_client()
    application = SimpleNamespace(id="app-1", name="Local Executor")
    validate_client = AsyncMock(return_value=application)
    refresh_tokens = AsyncMock(
        return_value=(_issued_token("new-access"), _issued_token("new-refresh"))
    )

    with (
        patch(
            "backend.api.features.oauth.validate_client_credentials",
            validate_client,
        ),
        patch("backend.api.features.oauth.refresh_tokens", refresh_tokens),
    ):
        response = client.post(
            "/api/oauth/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": "old-refresh",
                "client_id": "autogpt-local-executor",
            },
        )

    assert response.status_code == 200
    validate_client.assert_awaited_once_with(
        "autogpt-local-executor",
        "",
        allow_public_without_secret=True,
    )
    refresh_tokens.assert_awaited_once_with("old-refresh", "app-1")


@pytest.mark.parametrize("encoding", ["form", "json"])
def test_invalid_token_request_never_echoes_or_logs_credentials(
    encoding: str, caplog: pytest.LogCaptureFixture
) -> None:
    client = _make_client()
    sentinels = {
        "grant_type": "sentinel-grant-secret",
        "refresh_token": "sentinel-refresh-secret",
        "client_secret": "sentinel-client-secret",
        "code_verifier": "sentinel-verifier-secret",
        "client_id": "sentinel-client-id",
    }

    if encoding == "form":
        response = client.post("/api/oauth/token", data=sentinels)
    else:
        response = client.post("/api/oauth/token", json=sentinels)

    assert response.status_code == 422
    for sentinel in sentinels.values():
        assert sentinel not in response.text
        assert sentinel not in caplog.text


@pytest.mark.asyncio
async def test_invalid_token_request_severs_sensitive_validation_context() -> None:
    sentinel = "sentinel-refresh-secret"
    body = json.dumps(
        {
            "grant_type": "invalid-grant",
            "refresh_token": sentinel,
            "client_secret": "sentinel-client-secret",
        }
    ).encode()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/oauth/token",
            "headers": [(b"content-type", b"application/json")],
        },
        receive,
    )

    with pytest.raises(RequestValidationError) as exc_info:
        await oauth._parse_token_request(request)

    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert sentinel not in repr(exc_info.value)


def test_successful_revocation_pushes_connected_shim_shutdown() -> None:
    client = _make_client()
    application = SimpleNamespace(
        id="app-1", name="Local Executor", client_id="autogpt-local-executor"
    )
    push_revocation = AsyncMock()

    with (
        patch(
            "backend.api.features.oauth.validate_client_credentials",
            AsyncMock(return_value=application),
        ),
        patch(
            "backend.api.features.oauth.revoke_access_token",
            AsyncMock(return_value=SimpleNamespace(user_id="user-1")),
        ),
        patch("backend.api.features.oauth._push_shim_revocation", push_revocation),
    ):
        response = client.post(
            "/api/oauth/revoke",
            json={
                "token": "access-token",
                "token_type_hint": "access_token",
                "client_id": "autogpt-local-executor",
                "client_secret": "confidential-secret",
            },
        )

    assert response.status_code == 200
    push_revocation.assert_awaited_once_with("user-1", "autogpt-local-executor")


def test_public_client_can_revoke_with_presented_token_and_no_secret() -> None:
    client = _make_client()
    application = SimpleNamespace(
        id="app-1", name="Local Executor", client_id="autogpt-local-executor"
    )
    validate_client = AsyncMock(return_value=application)
    revoke_token = AsyncMock(return_value=SimpleNamespace(user_id="user-1"))

    with (
        patch(
            "backend.api.features.oauth.validate_client_credentials",
            validate_client,
        ),
        patch("backend.api.features.oauth.revoke_access_token", revoke_token),
        patch("backend.api.features.oauth._push_shim_revocation", AsyncMock()),
    ):
        response = client.post(
            "/api/oauth/revoke",
            json={
                "token": "access-token",
                "token_type_hint": "access_token",
                "client_id": "autogpt-local-executor",
            },
        )

    assert response.status_code == 200
    validate_client.assert_awaited_once_with(
        "autogpt-local-executor", "", allow_public_without_secret=True
    )
    revoke_token.assert_awaited_once_with("access-token", "app-1")


def test_revocation_storage_failure_returns_server_error() -> None:
    client = _make_client(raise_server_exceptions=False)
    application = SimpleNamespace(
        id="app-1", name="Local Executor", client_id="autogpt-local-executor"
    )
    push_revocation = AsyncMock()

    with (
        patch(
            "backend.api.features.oauth.validate_client_credentials",
            AsyncMock(return_value=application),
        ),
        patch(
            "backend.api.features.oauth.revoke_access_token",
            AsyncMock(side_effect=ConnectionError("database unavailable")),
        ),
        patch("backend.api.features.oauth._push_shim_revocation", push_revocation),
    ):
        response = client.post(
            "/api/oauth/revoke",
            json={
                "token": "access-token",
                "token_type_hint": "access_token",
                "client_id": "autogpt-local-executor",
            },
        )

    assert response.status_code == 500
    push_revocation.assert_not_awaited()


def test_revocation_push_failure_does_not_change_oauth_result() -> None:
    client = _make_client(raise_server_exceptions=False)
    application = SimpleNamespace(
        id="app-1", name="Local Executor", client_id="autogpt-local-executor"
    )
    revoke_token = AsyncMock(return_value=SimpleNamespace(user_id="user-1"))
    manager = MagicMock()
    manager.revoke_user_shims = AsyncMock(
        side_effect=[ConnectionError("relay unavailable"), 0]
    )
    payload = {
        "token": "access-token",
        "token_type_hint": "access_token",
        "client_id": "autogpt-local-executor",
    }

    with (
        patch(
            "backend.api.features.oauth.validate_client_credentials",
            AsyncMock(return_value=application),
        ),
        patch("backend.api.features.oauth.revoke_access_token", revoke_token),
        patch("backend.api.features.oauth.get_shim_manager", return_value=manager),
    ):
        first_response = client.post("/api/oauth/revoke", json=payload)
        retry_response = client.post("/api/oauth/revoke", json=payload)

    assert first_response.status_code == 200
    assert retry_response.status_code == 200
    assert revoke_token.await_count == 2
    assert manager.revoke_user_shims.await_count == 2


def test_direct_revocation_delivery_failure_does_not_change_oauth_result() -> None:
    client = _make_client(raise_server_exceptions=False)
    application = SimpleNamespace(
        id="app-1", name="Local Executor", client_id="autogpt-local-executor"
    )
    relay = _NoopRelay()
    manager = ShimConnectionManager(relay=cast(RelayBackend, relay))
    websocket = _RetryingDirectWebSocket()
    manager.register(
        "session-1",
        websocket,
        user_id="user-1",
        client_id="autogpt-local-executor",
    )
    payload = {
        "token": "access-token",
        "token_type_hint": "access_token",
        "client_id": "autogpt-local-executor",
    }

    with (
        patch(
            "backend.api.features.oauth.validate_client_credentials",
            AsyncMock(return_value=application),
        ),
        patch(
            "backend.api.features.oauth.revoke_access_token",
            AsyncMock(return_value=SimpleNamespace(user_id="user-1")),
        ),
        patch("backend.api.features.oauth.get_shim_manager", return_value=manager),
    ):
        first_response = client.post("/api/oauth/revoke", json=payload)
        retry_response = client.post("/api/oauth/revoke", json=payload)

    assert first_response.status_code == 200
    assert retry_response.status_code == 200
    assert websocket.send_attempts == 2
    assert websocket.close_attempts == 2
    assert relay.revocations == 2
    assert manager.get("session-1") is None


def test_refresh_replay_push_failure_does_not_change_oauth_result() -> None:
    client = _make_client(raise_server_exceptions=False)
    application = SimpleNamespace(
        id="app-1", name="Local Executor", client_id="autogpt-local-executor"
    )

    def refresh_error():
        return RefreshTokenFamilyRevokedError(
            "refresh token family has been revoked", "user-1", "app-1"
        )

    refresh = AsyncMock(side_effect=[refresh_error(), refresh_error()])
    manager = MagicMock()
    manager.revoke_user_shims = AsyncMock(
        side_effect=[ConnectionError("relay unavailable"), 0]
    )
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": "replayed-refresh-token",
        "client_id": "autogpt-local-executor",
    }

    with (
        patch(
            "backend.api.features.oauth.validate_client_credentials",
            AsyncMock(return_value=application),
        ),
        patch("backend.api.features.oauth.refresh_tokens", refresh),
        patch("backend.api.features.oauth.get_shim_manager", return_value=manager),
    ):
        first_response = client.post("/api/oauth/token", json=payload)
        retry_response = client.post("/api/oauth/token", json=payload)

    assert first_response.status_code == 400
    assert retry_response.status_code == 400
    assert "family has been revoked" in retry_response.json()["detail"]
    assert refresh.await_count == 2
    assert manager.revoke_user_shims.await_count == 2
