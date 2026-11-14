from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any
from unittest.mock import AsyncMock, Mock

import jwt
import pytest
from autogpt_libs.auth import closed_accounts
from autogpt_libs.auth import config as auth_config
from autogpt_libs.auth import jwt_utils
from autogpt_libs.auth.service import (
    FRONTEND_SERVICE_SUBJECT,
    SERVICE_TOKEN_AUDIENCE,
    requires_frontend_service,
)
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import FastAPI, HTTPException, Security, WebSocket
from fastapi.testclient import TestClient
from jwt.algorithms import ECAlgorithm

from backend.api import session_auth, ws_api
from backend.util.settings import Config


async def test_rollout_default_accepts_preexisting_signed_tokens(
    client: TestClient,
    signing_key: ec.EllipticCurvePrivateKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        session_auth.settings.config,
        "auth_legacy_token_deadline",
        Config.model_fields["auth_legacy_token_deadline"].default,
    )
    token = _sign(signing_key, {"sub": "user", "aud": "authenticated"})
    assert (
        client.get("/user", headers={"Authorization": f"Bearer {token}"}).status_code
        == 200
    )


async def test_sidless_user_token_requires_a_migration_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_auth.settings.config, "auth_legacy_token_deadline", 0)
    with pytest.raises(HTTPException) as rejected:
        await session_auth.require_active_auth_session(
            {"sub": "user", "aud": "authenticated"}
        )
    assert rejected.value.status_code == 401


@pytest.fixture
def signing_key(monkeypatch: pytest.MonkeyPatch) -> ec.EllipticCurvePrivateKey:
    key = ec.generate_private_key(ec.SECP256R1())
    public = ECAlgorithm.to_jwk(key.public_key(), as_dict=True)
    public.update({"kid": "migration-test", "alg": "ES256", "use": "sig"})
    monkeypatch.setenv("JWT_JWKS_URL", "http://localhost/migration-test/jwks")
    monkeypatch.setattr(auth_config, "_settings", auth_config.Settings())
    monkeypatch.setattr(jwt_utils, "_jwks_client", None)
    monkeypatch.setattr(jwt_utils, "_jwks_client_url", None)
    monkeypatch.setattr(jwt.PyJWKClient, "fetch_data", lambda _: {"keys": [public]})
    return key


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = FastAPI()
    app.state.jwt_payload_validator = session_auth.require_active_auth_session

    @app.get("/user")
    async def user(
        payload: Annotated[dict[str, Any], Security(jwt_utils.get_jwt_payload)],
    ) -> dict[str, str]:
        return {"sub": payload["sub"]}

    @app.get(
        "/service",
        dependencies=[Security(requires_frontend_service("auth-email:send"))],
    )
    async def service() -> dict[str, bool]:
        return {"ok": True}

    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize(
    "deadline,expected_status", [(0, 401), (99, 401), (100, 401), (101, 200)]
)
async def test_signed_legacy_token_cutoff_applies_to_http_and_websocket(
    client: TestClient,
    signing_key: ec.EllipticCurvePrivateKey,
    monkeypatch: pytest.MonkeyPatch,
    deadline: int,
    expected_status: int,
) -> None:
    monkeypatch.setattr(session_auth, "time", lambda: 100)
    monkeypatch.setattr(
        session_auth.settings.config, "auth_legacy_token_deadline", deadline
    )
    monkeypatch.setattr(ws_api, "is_user_banned", AsyncMock(return_value=False))
    accounts = Mock(is_account_closed=AsyncMock(return_value=False))
    monkeypatch.setattr(ws_api, "user_db", Mock(return_value=accounts))
    token = _sign(signing_key, {"sub": "user", "aud": "authenticated"})

    response = client.get("/user", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == expected_status
    send = AsyncMock()
    websocket = WebSocket(
        {"type": "websocket", "query_string": f"token={token}".encode()},
        receive=AsyncMock(return_value={"type": "websocket.connect"}),
        send=send,
    )
    identity = await ws_api.authenticate_websocket(websocket)
    if expected_status == 200:
        assert identity == "user"
        send.assert_not_awaited()
    else:
        assert identity == ""
        assert send.await_args is not None
        assert send.await_args.args[0]["code"] == 1013
        assert response.headers["X-Error-Code"] == "session_token_renewal_required"


def test_service_token_remains_separate_from_user_cutoff(
    client: TestClient,
    signing_key: ec.EllipticCurvePrivateKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_auth.settings.config, "auth_legacy_token_deadline", 0)
    token = _sign(
        signing_key,
        {
            "sub": FRONTEND_SERVICE_SUBJECT,
            "aud": SERVICE_TOKEN_AUDIENCE,
            "scope": "auth-email:send",
        },
    )
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/service", headers=headers).status_code == 200
    assert client.get("/user", headers=headers).status_code == 401


@pytest.mark.parametrize("sid", [None, "", 1, False])
async def test_migration_window_does_not_allow_malformed_session_claims(
    monkeypatch: pytest.MonkeyPatch, sid: Any
) -> None:
    monkeypatch.setattr(session_auth, "time", lambda: 100)
    monkeypatch.setattr(session_auth.settings.config, "auth_legacy_token_deadline", 101)
    with pytest.raises(HTTPException) as rejected:
        await session_auth.require_active_auth_session({"sub": "user", "sid": sid})
    assert rejected.value.status_code == 401


@pytest.fixture
def ended_session(monkeypatch: pytest.MonkeyPatch) -> Iterator[AsyncMock]:
    """A token whose session is gone, for an account that isn't banned."""
    monkeypatch.setattr(
        session_auth.user_store, "is_user_banned", AsyncMock(return_value=False)
    )
    sessions = Mock(is_auth_session_active=AsyncMock(return_value=False))
    monkeypatch.setattr(session_auth, "auth_session_db", Mock(return_value=sessions))
    users = Mock(is_auth_user_banned=AsyncMock(return_value=False))
    monkeypatch.setattr(session_auth, "user_db", Mock(return_value=users))
    closed = AsyncMock(return_value=False)
    monkeypatch.setattr(closed_accounts, "_closed_account_query", lambda: closed)
    closed_accounts.forget_account_state("user")
    yield closed
    closed_accounts.forget_account_state("user")


async def test_an_ended_session_is_refused_with_a_code_the_client_signs_out_on(
    ended_session: AsyncMock,
) -> None:
    with pytest.raises(HTTPException) as rejected:
        await session_auth.require_active_auth_session({"sub": "user", "sid": "s1"})
    assert rejected.value.status_code == 401
    assert rejected.value.detail == {
        "code": "session_revoked",
        "message": "Session expired or revoked",
    }


async def test_a_closed_account_is_refused_as_closed_not_as_an_ended_session(
    ended_session: AsyncMock,
) -> None:
    # Closing an account deletes every session, so its tokens reach the
    # "not active" answer.
    ended_session.return_value = True
    with pytest.raises(HTTPException) as rejected:
        await session_auth.require_active_auth_session({"sub": "user", "sid": "s1"})
    assert rejected.value.status_code == 401
    assert rejected.value.detail == closed_accounts.ACCOUNT_CLOSED_DETAIL


async def test_a_live_session_costs_no_closed_account_lookup(
    ended_session: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions = Mock(is_auth_session_active=AsyncMock(return_value=True))
    monkeypatch.setattr(session_auth, "auth_session_db", Mock(return_value=sessions))
    await session_auth.require_active_auth_session({"sub": "user", "sid": "s1"})
    ended_session.assert_not_awaited()


def _sign(key: ec.EllipticCurvePrivateKey, payload: dict[str, Any]) -> str:
    return jwt.encode(
        {**payload, "exp": datetime.now(UTC) + timedelta(minutes=5)},
        key,
        algorithm="ES256",
        headers={"kid": "migration-test"},
    )
