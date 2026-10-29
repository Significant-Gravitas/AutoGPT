from datetime import UTC, datetime

import fastapi
import jwt
import pytest
from autogpt_libs.auth import config as auth_config
from autogpt_libs.auth import jwt_utils
from autogpt_libs.auth.service import FRONTEND_SERVICE_SUBJECT, SERVICE_TOKEN_AUDIENCE
from cryptography.hazmat.primitives.asymmetric import ec
from httpx import ASGITransport, AsyncClient
from jwt.algorithms import ECAlgorithm

from backend.api.utils.client_ip import CLIENT_IP_SCOPE, ClientIP

SOCKET_IP = "198.51.100.1"
# Every header a proxy might use for the client, set by the caller itself.
CLAIMED_ADDRESS = {
    "CF-Connecting-IP": "192.0.2.10",
    "X-Forwarded-For": "192.0.2.10",
    "X-Real-IP": "192.0.2.10",
    "True-Client-IP": "192.0.2.10",
}

app = fastapi.FastAPI()


@app.get("/client")
async def _client(client: ClientIP) -> str:
    return client


@pytest.fixture
def frontend_key(monkeypatch):
    """The frontend's JWKS signing key, as the backend would fetch it."""
    key = ec.generate_private_key(ec.SECP256R1())
    jwk = ECAlgorithm.to_jwk(key.public_key(), as_dict=True)
    jwk.update({"kid": "frontend-1", "alg": "ES256", "use": "sig"})
    monkeypatch.setenv("JWT_JWKS_URL", "https://platform.example.com/api/auth/jwks")
    monkeypatch.setattr(auth_config, "_settings", auth_config.Settings())
    monkeypatch.setattr(jwt_utils, "_jwks_client", None)
    monkeypatch.setattr(jwt_utils, "_jwks_client_url", None)
    monkeypatch.setattr(jwt.PyJWKClient, "fetch_data", lambda self: {"keys": [jwk]})
    return key


@pytest.mark.asyncio
async def test_the_ip_the_frontend_signed_is_the_client(frontend_key):
    token = _ip_token(frontend_key, "203.0.113.7")
    headers = {"X-Client-IP-Token": token, **CLAIMED_ADDRESS}

    assert await _client_of(headers) == "203.0.113.7"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "forge",
    [
        lambda key: None,
        lambda key: _ip_token(ec.generate_private_key(ec.SECP256R1()), "203.0.113.7"),
        lambda key: _ip_token(key, "203.0.113.7", scope="client-country"),
        lambda key: _ip_token(key, "203.0.113.7", sub="user-1", aud="authenticated"),
        lambda key: _ip_token(key, "203.0.113.7", exp=1),
        lambda key: jwt.encode(
            {
                "sub": FRONTEND_SERVICE_SUBJECT,
                "aud": SERVICE_TOKEN_AUDIENCE,
                "scope": CLIENT_IP_SCOPE,
                "ip": "203.0.113.7",
            },
            "a-shared-secret-anyone-could-guess-0123456789",
            algorithm="HS256",
        ),
        lambda key: _ip_token(key, ["203.0.113.7"]),
    ],
    ids=[
        "missing",
        "foreign-key",
        "other-scope",
        "user-token",
        "expired",
        "hs256",
        "non-string",
    ],
)
async def test_without_a_token_the_frontend_issued_the_caller_is_its_own_address(
    frontend_key, forge
):
    headers = dict(CLAIMED_ADDRESS)
    if token := forge(frontend_key):
        headers["X-Client-IP-Token"] = token

    assert await _client_of(headers) == SOCKET_IP


def _ip_token(key, ip, **overrides) -> str:
    now = int(datetime.now(UTC).timestamp())
    claims = {
        "sub": FRONTEND_SERVICE_SUBJECT,
        "aud": SERVICE_TOKEN_AUDIENCE,
        "scope": CLIENT_IP_SCOPE,
        "ip": ip,
        "iat": now,
        "exp": now + 60,
        **overrides,
    }
    return jwt.encode(claims, key, algorithm="ES256", headers={"kid": "frontend-1"})


async def _client_of(headers: dict[str, str]) -> str:
    transport = ASGITransport(app=app, client=(SOCKET_IP, 4321))
    async with AsyncClient(transport=transport, base_url="https://example.com") as c:
        response = await c.get("/client", headers=headers)
    assert response.status_code == 200
    return response.json()
