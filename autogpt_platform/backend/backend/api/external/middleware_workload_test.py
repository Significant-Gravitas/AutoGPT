"""The External API accepts a workload token as a bearer credential."""

from datetime import datetime, timedelta, timezone

import fastapi
import pytest
from fastapi.testclient import TestClient
from prisma.enums import APIKeyPermission

from backend.api.external import middleware
from backend.data.auth.workload_identity import (
    InvalidWorkloadTokenError,
    WorkloadTokenInfo,
)

app = fastapi.FastAPI()


@app.get("/auth")
async def by_auth(auth=fastapi.Security(middleware.require_auth)):
    return {"user_id": auth.user_id, "type": auth.type}


@app.get("/read-graph")
async def by_permission(
    auth=fastapi.Security(middleware.require_permission(APIKeyPermission.READ_GRAPH)),
):
    return {"user_id": auth.user_id}


client = TestClient(app)
HEADERS = {"Authorization": "Bearer agpt_wt_token"}


@pytest.fixture
def owner(mocker):
    validate = mocker.patch.object(
        middleware, "validate_workload_token", return_value=_token_info()
    )
    access_token = mocker.patch.object(middleware, "validate_access_token")
    banned = mocker.patch.object(middleware, "is_user_banned", return_value=False)
    return validate, access_token, banned


def test_a_workload_token_authenticates_as_the_identitys_owner(owner):
    validate, access_token, _ = owner

    response = client.get("/auth", headers=HEADERS)

    assert response.status_code == 200
    assert response.json() == {"user_id": "user-1", "type": "workload"}
    validate.assert_awaited_once_with("agpt_wt_token")
    access_token.assert_not_called()


def test_a_workload_tokens_scopes_gate_the_endpoint(owner):
    owner[0].return_value = _token_info(scopes=[APIKeyPermission.IDENTITY])

    response = client.get("/read-graph", headers=HEADERS)

    assert response.status_code == 403


def test_an_invalid_workload_token_is_unauthorized(owner):
    owner[0].side_effect = InvalidWorkloadTokenError("Workload token expired")

    response = client.get("/auth", headers=HEADERS)

    assert response.status_code == 401
    assert response.json()["detail"] == "Workload token expired"


def test_a_banned_owners_workload_token_is_refused(owner):
    owner[2].return_value = True

    response = client.get("/auth", headers=HEADERS)

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "account_banned"


def test_other_bearer_tokens_still_go_to_oauth(owner):
    validate, access_token, _ = owner
    access_token.side_effect = middleware.InvalidTokenError("access token not found")

    response = client.get("/auth", headers={"Authorization": "Bearer agpt_xt_token"})

    assert response.status_code == 401
    access_token.assert_awaited_once()
    validate.assert_not_called()


def _token_info(scopes=None) -> WorkloadTokenInfo:
    now = datetime.now(timezone.utc)
    return WorkloadTokenInfo(
        id="wt-1",
        type="workload",
        user_id="user-1",
        scopes=scopes or [APIKeyPermission.READ_GRAPH],
        created_at=now,
        expires_at=now + timedelta(minutes=15),
        workload_identity_id="wi-1",
        subject="repo:acme/api:ref:refs/heads/main",
    )
