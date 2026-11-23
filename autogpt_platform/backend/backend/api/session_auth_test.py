from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any
from unittest.mock import AsyncMock, Mock

import jwt
import pytest
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


async def test_a_signed_token_without_sid_must_be_renewed_on_http_and_websocket(
    client: TestClient,
    signing_key: ec.EllipticCurvePrivateKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user token that names no session is refused, never trusted on its
    signature and expiry alone, so signing out or a ban always reaches it."""
    monkeypatch.setattr(ws_api, "is_user_banned", AsyncMock(return_value=False))
    accounts = Mock(is_account_closed=AsyncMock(return_value=False))
    monkeypatch.setattr(ws_api, "user_db", Mock(return_value=accounts))
    token = _sign(signing_key, {"sub": "user", "aud": "authenticated"})

    response = client.get("/user", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
    assert response.headers["X-Error-Code"] == "session_token_renewal_required"

    send = AsyncMock()
    websocket = WebSocket(
        {"type": "websocket", "query_string": f"token={token}".encode()},
        receive=AsyncMock(return_value={"type": "websocket.connect"}),
        send=send,
    )
    assert await ws_api.authenticate_websocket(websocket) is None
    assert send.await_args is not None
    closed = send.await_args.args[0]
    assert (closed["code"], closed["reason"]) == (
        1013,
        "Session token requires renewal",
    )


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


def test_service_token_remains_separate_from_the_user_session_check(
    client: TestClient,
    signing_key: ec.EllipticCurvePrivateKey,
) -> None:
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
async def test_malformed_session_claims_are_refused(sid: Any) -> None:
    with pytest.raises(HTTPException) as rejected:
        await session_auth.require_active_auth_session({"sub": "user", "sid": sid})
    assert rejected.value.status_code == 401


def _sign(key: ec.EllipticCurvePrivateKey, payload: dict[str, Any]) -> str:
    return jwt.encode(
        {**payload, "exp": datetime.now(UTC) + timedelta(minutes=5)},
        key,
        algorithm="ES256",
        headers={"kid": "migration-test"},
    )
