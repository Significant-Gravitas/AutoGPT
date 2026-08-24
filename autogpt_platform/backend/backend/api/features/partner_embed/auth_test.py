import os
from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
import pytest
from autogpt_libs.auth import config, jwt_utils
from autogpt_libs.auth.config import Settings
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import FastAPI, Security
from fastapi.testclient import TestClient
from jwt.algorithms import ECAlgorithm
from pytest_mock import MockerFixture

from backend.api.features.partner_embed.auth import (
    EMBED_TOKEN_AUDIENCE,
    EmbedPrincipal,
    requires_embed_principal,
)

MOCK_JWKS_URL = "http://localhost:3000/api/auth/jwks"

app = FastAPI()


@app.get("/guarded")
async def guarded_endpoint(
    principal: Annotated[EmbedPrincipal, Security(requires_embed_principal)],
) -> EmbedPrincipal:
    return principal


client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup():
    """This pure auth suite does not need the repository integration server."""
    yield


def make_es256_keypair(kid: str = "embed-test-key"):
    private_key = ec.generate_private_key(ec.SECP256R1())
    jwk = ECAlgorithm.to_jwk(private_key.public_key(), as_dict=True)
    jwk.update({"kid": kid, "alg": "ES256", "use": "sig"})
    return private_key, {"keys": [jwk]}


def create_token(payload: dict, private_key, kid: str = "embed-test-key") -> str:
    return jwt.encode(payload, private_key, algorithm="ES256", headers={"kid": kid})


@pytest.fixture
def jwks_config(mocker: MockerFixture):
    mocker.patch.dict(os.environ, {"JWT_JWKS_URL": MOCK_JWKS_URL}, clear=True)
    mocker.patch.object(config, "_settings", Settings())
    mocker.patch.object(jwt_utils, "_jwks_client", None)
    mocker.patch.object(jwt_utils, "_jwks_client_url", None)
    private_key, jwk_set = make_es256_keypair()
    mocker.patch.object(jwt.PyJWKClient, "fetch_data", return_value=jwk_set)
    return private_key


def embed_payload() -> dict:
    return {
        "sub": "0234dc86-e049-5c61-8b7e-826f7a7c225f",
        "aud": EMBED_TOKEN_AUDIENCE,
        "token_use": "partner_embed",
        "partner_id": "forwarding-digital",
        "organization_id": "70d89c3b-2af3-5f56-8a21-2951b469ba95",
        "team_id": "600e3708-3a7a-54c7-b527-53d2c62b8d5b",
        "external_account_id": "forwarder-42",
        "scope": "embed:chat embed:schedules",
        "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
    }


def get_guarded(token: str):
    return client.get("/guarded", headers={"Authorization": f"Bearer {token}"})


def test_valid_embed_token_returns_locked_principal(jwks_config):
    response = get_guarded(create_token(embed_payload(), jwks_config))

    assert response.status_code == 200
    assert response.json() == {
        "user_id": "0234dc86-e049-5c61-8b7e-826f7a7c225f",
        "partner_id": "forwarding-digital",
        "organization_id": "70d89c3b-2af3-5f56-8a21-2951b469ba95",
        "team_id": "600e3708-3a7a-54c7-b527-53d2c62b8d5b",
        "external_account_id": "forwarder-42",
        "scopes": ["embed:chat", "embed:schedules"],
    }


def test_normal_user_token_cannot_open_embed_api(jwks_config):
    payload = {**embed_payload(), "aud": "authenticated"}
    response = get_guarded(create_token(payload, jwks_config))

    assert response.status_code == 401


@pytest.mark.parametrize(
    "claim", ["partner_id", "organization_id", "external_account_id"]
)
def test_missing_tenant_claim_is_rejected(jwks_config, claim: str):
    payload = embed_payload()
    del payload[claim]
    response = get_guarded(create_token(payload, jwks_config))

    assert response.status_code == 401


def test_symmetric_embed_token_is_rejected(jwks_config):
    payload = embed_payload()
    token = jwt.encode(payload, "a" * 32, algorithm="HS256")
    response = get_guarded(token)

    assert response.status_code == 401
    assert "symmetric" in response.json()["detail"]
