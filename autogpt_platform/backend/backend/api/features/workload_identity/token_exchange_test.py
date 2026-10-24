"""The token-exchange grant of POST /api/oauth/token needs no client secret."""

from datetime import datetime, timedelta, timezone

import fastapi
import pytest
from fastapi.testclient import TestClient
from prisma.enums import APIKeyPermission

from backend.api.features import oauth
from backend.api.features.workload_identity import token_exchange
from backend.data.auth.workload_exchange import (
    InvalidScopeError,
    InvalidSubjectTokenError,
)
from backend.data.auth.workload_identity import WorkloadTokenInfo

GRANT = "urn:ietf:params:oauth:grant-type:token-exchange"

app = fastapi.FastAPI()
app.include_router(oauth.router, prefix="/api/oauth")
client = TestClient(app)


def _body(**overrides) -> dict:
    return {
        "grant_type": GRANT,
        "subject_token": "eyJ.oidc.token",
        "subject_token_type": "urn:ietf:params:oauth:token-type:id_token",
        "client_id": "wi-1",
        **overrides,
    }


def test_a_trusted_workload_token_is_exchanged(mocker):
    exchange = mocker.patch.object(
        token_exchange,
        "exchange_for_workload_token",
        return_value=(_token_info(), "agpt_wt_secret"),
    )
    validate_client = mocker.patch.object(oauth, "validate_client_credentials")

    response = client.post("/api/oauth/token", json=_body(scope="READ_GRAPH"))

    assert response.status_code == 200
    body = response.json()
    assert body["access_token"] == "agpt_wt_secret"
    assert body["token_type"] == "Bearer"
    assert body["issued_token_type"] == "urn:ietf:params:oauth:token-type:access_token"
    assert 0 < body["expires_in"] <= 900
    assert body["scope"] == "READ_GRAPH"
    exchange.assert_awaited_once_with("wi-1", "eyJ.oidc.token", ["READ_GRAPH"])
    validate_client.assert_not_called()


@pytest.mark.parametrize(
    "error, code",
    [
        (InvalidSubjectTokenError("Token rejected"), "invalid_grant"),
        (InvalidScopeError("Not granted"), "invalid_scope"),
    ],
)
def test_a_refused_exchange_answers_with_the_oauth_error_code(mocker, error, code):
    mocker.patch.object(
        token_exchange, "exchange_for_workload_token", side_effect=error
    )

    response = client.post("/api/oauth/token", json=_body())

    assert response.status_code == 400
    assert response.json()["detail"] == {
        "error": code,
        "error_description": str(error),
    }


def test_an_unknown_subject_token_type_is_rejected(mocker):
    exchange = mocker.patch.object(token_exchange, "exchange_for_workload_token")

    response = client.post(
        "/api/oauth/token",
        json=_body(subject_token_type="urn:ietf:params:oauth:token-type:saml2"),
    )

    assert response.status_code == 422
    exchange.assert_not_called()


def test_the_code_grant_still_requires_the_client_secret(mocker):
    exchange = mocker.patch.object(token_exchange, "exchange_for_workload_token")

    response = client.post(
        "/api/oauth/token",
        json={
            "grant_type": "authorization_code",
            "code": "code",
            "redirect_uri": "https://app.example/cb",
            "client_id": "app",
            "code_verifier": "verifier",
        },
    )

    assert response.status_code == 422
    exchange.assert_not_called()


def _token_info() -> WorkloadTokenInfo:
    now = datetime.now(timezone.utc)
    return WorkloadTokenInfo(
        id="wt-1",
        type="workload",
        user_id="user-1",
        scopes=[APIKeyPermission.READ_GRAPH],
        created_at=now,
        expires_at=now + timedelta(seconds=900),
        workload_identity_id="wi-1",
        subject="repo:acme/api:ref:refs/heads/main",
    )
