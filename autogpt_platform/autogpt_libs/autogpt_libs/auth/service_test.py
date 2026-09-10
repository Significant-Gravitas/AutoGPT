"""
Tests for frontend service-token authentication (requires_frontend_service).

Service tokens ride the same JWKS trust as user tokens but with a distinct
audience and subject; these tests pin the separation between the two planes.
"""

import os
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import FastAPI, Security
from fastapi.testclient import TestClient
from jwt.algorithms import ECAlgorithm
from pytest_mock import MockerFixture

from autogpt_libs.auth import config, jwt_utils
from autogpt_libs.auth.config import Settings
from autogpt_libs.auth.service import (
    FRONTEND_SERVICE_SUBJECT,
    SERVICE_TOKEN_AUDIENCE,
    requires_frontend_service,
)

MOCK_JWT_SECRET = "test-secret-key-with-at-least-32-characters"
MOCK_JWKS_URL = "http://localhost:3000/api/auth/jwks"

SERVICE_PAYLOAD = {
    "sub": FRONTEND_SERVICE_SUBJECT,
    "aud": SERVICE_TOKEN_AUDIENCE,
    "scope": "auth-email:send",
}

app = FastAPI()


@app.post(
    "/guarded",
    dependencies=[Security(requires_frontend_service("auth-email:send"))],
)
def guarded_endpoint():
    return {"ok": True}


client = TestClient(app)


def make_es256_keypair(kid: str = "test-key-1"):
    private_key = ec.generate_private_key(ec.SECP256R1())
    jwk = ECAlgorithm.to_jwk(private_key.public_key(), as_dict=True)
    jwk.update({"kid": kid, "alg": "ES256", "use": "sig"})
    return private_key, {"keys": [jwk]}


def create_es256_token(payload, private_key, kid: str = "test-key-1") -> str:
    return jwt.encode(payload, private_key, algorithm="ES256", headers={"kid": kid})


@pytest.fixture
def jwks_config(mocker: MockerFixture):
    """Configure both the legacy shared secret and a JWKS endpoint."""
    mocker.patch.dict(
        os.environ,
        {"JWT_VERIFY_KEY": MOCK_JWT_SECRET, "JWT_JWKS_URL": MOCK_JWKS_URL},
        clear=True,
    )
    mocker.patch.object(config, "_settings", Settings())
    mocker.patch.object(jwt_utils, "_jwks_client", None)
    mocker.patch.object(jwt_utils, "_jwks_client_url", None)

    private_key, jwk_set = make_es256_keypair()
    mocker.patch.object(jwt.PyJWKClient, "fetch_data", return_value=jwk_set)
    yield private_key


def _post(token: str | None = None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.post("/guarded", headers=headers)


def test_valid_service_token_is_accepted(jwks_config):
    token = create_es256_token(SERVICE_PAYLOAD, jwks_config)
    response = _post(token)
    assert response.status_code == 200


def test_scope_list_containing_required_scope_is_accepted(jwks_config):
    payload = {**SERVICE_PAYLOAD, "scope": "other:scope auth-email:send"}
    token = create_es256_token(payload, jwks_config)
    assert _post(token).status_code == 200


def test_missing_authorization_header_is_401(jwks_config):
    assert _post().status_code == 401


def test_user_token_is_rejected_as_service_token(jwks_config):
    """A user token (aud=authenticated) must not open service endpoints."""
    user_payload = {"sub": "user-123", "aud": "authenticated", "role": "admin"}
    token = create_es256_token(user_payload, jwks_config)
    assert _post(token).status_code == 401


def test_wrong_subject_is_401(jwks_config):
    payload = {**SERVICE_PAYLOAD, "sub": "service:imposter"}
    token = create_es256_token(payload, jwks_config)
    assert _post(token).status_code == 401


def test_missing_scope_is_403(jwks_config):
    payload = {**SERVICE_PAYLOAD, "scope": "other:scope"}
    token = create_es256_token(payload, jwks_config)
    assert _post(token).status_code == 403


def test_symmetric_service_token_is_rejected(jwks_config):
    """The legacy HS256 shared secret must not be able to mint service tokens."""
    token = jwt.encode(SERVICE_PAYLOAD, MOCK_JWT_SECRET, algorithm="HS256")
    response = _post(token)
    assert response.status_code == 401
    assert "symmetric" in response.json()["detail"]


def test_expired_service_token_is_401(jwks_config):
    payload = {
        **SERVICE_PAYLOAD,
        "exp": datetime.now(timezone.utc) - timedelta(minutes=1),
    }
    token = create_es256_token(payload, jwks_config)
    assert _post(token).status_code == 401


def test_garbage_token_is_401(jwks_config):
    assert _post("not-a-jwt").status_code == 401


# requires_frontend_service's "503 when JWT_JWKS_URL is unset" guard has no
# test here: a valid Settings() cannot exist without JWT_JWKS_URL (its
# validate() raises), so the guard is unreachable. config_test.py covers that
# enforcement.
