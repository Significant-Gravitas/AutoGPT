"""Managing workload identities: validation, ownership scoping, revocation."""

from datetime import datetime, timezone

import fastapi
import pytest
from autogpt_libs.auth.dependencies import get_request_context
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from fastapi.testclient import TestClient
from prisma.enums import APIKeyPermission, WorkloadIdentityStatus

from backend.api.features.workload_identity import routes
from backend.data.auth.workload_identity import WorkloadIdentityInfo
from backend.data.auth.workload_trust import ClaimCondition

app = fastapi.FastAPI()
app.include_router(routes.router, prefix="/api/workload-identities")
client = TestClient(app)

SPEC = {
    "name": "deploy from CI",
    "issuer": "https://token.actions.githubusercontent.com",
    "audience": "https://backend.agpt.co",
    "claim_conditions": [{"claim": "repository", "pattern": "acme/api"}],
    "permissions": ["EXECUTE_GRAPH"],
}


@pytest.fixture(autouse=True)
def auth(mock_jwt_user):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    app.dependency_overrides[get_request_context] = mock_jwt_user["get_request_context"]
    yield
    app.dependency_overrides.clear()


def test_creating_an_identity_scopes_it_to_the_caller_and_org(mocker, test_user_id):
    create = mocker.patch.object(
        routes.workload_identity_db,
        "create_workload_identity",
        return_value=_info(),
    )

    response = client.post("/api/workload-identities", json=SPEC)

    assert response.status_code == 200
    assert response.json()["id"] == "wi-1"
    user_id, spec = create.await_args.args
    assert user_id == test_user_id
    assert spec.issuer == SPEC["issuer"]
    assert create.await_args.kwargs == {"organization_id": "test-org"}


@pytest.mark.parametrize(
    "overrides",
    [
        {"claim_conditions": [{"claim": "ref", "pattern": "refs/heads/main"}]},
        {"claim_conditions": []},
        {"issuer": "http://token.actions.githubusercontent.com"},
        {"permissions": []},
        {"token_ttl_seconds": 86400},
    ],
    ids=["unpinned", "no conditions", "http issuer", "no permissions", "ttl"],
)
def test_an_identity_that_would_trust_too_much_is_rejected(mocker, overrides):
    create = mocker.patch.object(
        routes.workload_identity_db, "create_workload_identity"
    )

    response = client.post("/api/workload-identities", json={**SPEC, **overrides})

    assert response.status_code == 422
    create.assert_not_called()


def test_an_identity_can_be_disabled(mocker, test_user_id):
    set_status = mocker.patch.object(
        routes.workload_identity_db,
        "set_workload_identity_status",
        return_value=_info(status=WorkloadIdentityStatus.DISABLED),
    )

    response = client.patch(
        "/api/workload-identities/wi-1", json={"status": "DISABLED"}
    )

    assert response.status_code == 200
    assert set_status.await_args.args == (
        "wi-1",
        test_user_id,
        WorkloadIdentityStatus.DISABLED,
    )


def test_revocation_is_not_a_status_patch(mocker):
    set_status = mocker.patch.object(
        routes.workload_identity_db, "set_workload_identity_status"
    )

    response = client.patch("/api/workload-identities/wi-1", json={"status": "REVOKED"})

    assert response.status_code == 422
    set_status.assert_not_called()


def test_deleting_an_identity_revokes_it(mocker, test_user_id):
    revoke = mocker.patch.object(
        routes.workload_identity_db,
        "revoke_workload_identity",
        return_value=_info(status=WorkloadIdentityStatus.REVOKED),
    )

    response = client.delete("/api/workload-identities/wi-1")

    assert response.status_code == 200
    assert response.json()["status"] == "REVOKED"
    revoke.assert_awaited_once_with("wi-1", test_user_id, organization_id="test-org")


def _info(status=WorkloadIdentityStatus.ACTIVE) -> WorkloadIdentityInfo:
    return WorkloadIdentityInfo(
        id="wi-1",
        name="deploy from CI",
        issuer=SPEC["issuer"],
        audience=SPEC["audience"],
        claim_conditions=[ClaimCondition(claim="repository", pattern="acme/api")],
        permissions=[APIKeyPermission.EXECUTE_GRAPH],
        token_ttl_seconds=900,
        status=status,
        created_at=datetime.now(timezone.utc),
        user_id="test-user-id",
    )
