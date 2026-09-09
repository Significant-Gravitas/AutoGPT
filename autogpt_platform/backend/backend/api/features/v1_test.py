import json
from datetime import datetime, timezone
from unittest.mock import Mock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from autogpt_libs.auth.models import RequestContext
from pytest_snapshot.plugin import Snapshot

from backend.api.rest_api import handle_internal_http_error
from backend.integrations.webhooks.graph_lifecycle_hooks import GraphActivationError

from .v1 import v1_router


def _test_ctx(user_id: str) -> RequestContext:
    return RequestContext(
        user_id=user_id,
        org_id="test-org",
        team_id="test-workspace",
        is_org_owner=True,
        is_org_admin=True,
        is_org_billing_manager=False,
        is_team_admin=True,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )


app = fastapi.FastAPI()
app.include_router(v1_router)
# Mirror rest_api.py's GraphActivationError → 400 mapping so the atomicity
# tests below verify the same behaviour the real app exposes.
app.add_exception_handler(GraphActivationError, handle_internal_http_error(400))

client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user, setup_test_user, test_user_id):
    """Setup auth overrides for all tests in this module"""
    from autogpt_libs.auth.dependencies import get_request_context
    from autogpt_libs.auth.jwt_utils import get_jwt_payload
    from autogpt_libs.auth.models import RequestContext

    # setup_test_user fixture already executed and user is created in database
    # It returns the user_id which we don't need to await

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]

    # Override get_request_context too — the real one queries Prisma to
    # resolve the user's personal org when no X-Org-Id header is set,
    # which closes/leaks the test event loop across sync TestClient calls.
    async def _fake_request_context() -> RequestContext:
        return RequestContext(
            user_id=test_user_id,
            org_id="test-org",
            team_id=None,
            is_org_owner=True,
            is_org_admin=True,
            is_org_billing_manager=False,
            is_team_admin=True,
            is_team_billing_manager=False,
            seat_status="ACTIVE",
        )

    app.dependency_overrides[get_request_context] = _fake_request_context
    yield
    app.dependency_overrides.clear()


# Auth endpoints tests
def test_get_or_create_user_route(
    mocker: pytest_mock.MockFixture,
    configured_snapshot: Snapshot,
    test_user_id: str,
) -> None:
    """Test get or create user endpoint"""
    mock_user = Mock()
    mock_user.created_at = datetime.now(timezone.utc)
    mock_user.model_dump.return_value = {
        "id": test_user_id,
        "email": "test@example.com",
        "name": "Test User",
    }
    mock_result = Mock(user=mock_user, was_created=False)

    mocker.patch(
        "backend.api.features.v1.get_or_create_user_with_status",
        return_value=mock_result,
    )

    response = client.post("/auth/user")

    assert response.status_code == 200
    assert response.headers["X-AutoGPT-User-Created"] == "false"
    response_data = response.json()

    configured_snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "auth_user",
    )


def test_get_or_create_user_route_reports_creation(
    mocker: pytest_mock.MockFixture,
    test_user_id: str,
) -> None:
    mock_user = Mock()
    mock_user.created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    mock_user.model_dump.return_value = {
        "id": test_user_id,
        "email": "test@example.com",
    }

    mocker.patch(
        "backend.api.features.v1.get_or_create_user_with_status",
        return_value=Mock(user=mock_user, was_created=True),
    )

    response = client.post("/auth/user")

    assert response.status_code == 200
    assert response.headers["X-AutoGPT-User-Created"] == "true"


def test_get_or_create_user_route_documents_creation_header() -> None:
    response_schema = app.openapi()["paths"]["/auth/user"]["post"]["responses"]["200"]

    assert response_schema["headers"]["X-AutoGPT-User-Created"] == {
        "description": "Whether this request created a new user",
        "schema": {"type": "string", "enum": ["true", "false"]},
    }


def test_update_user_email_route(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    """Test update user email endpoint"""
    mocker.patch(
        "backend.api.features.v1.update_user_email",
        return_value=None,
    )

    response = client.post("/auth/user/email", json="newemail@example.com")

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["email"] == "newemail@example.com"

    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "auth_email",
    )


# Invalid request tests
def test_invalid_json_request() -> None:
    """Test endpoint with invalid JSON"""
    response = client.post(
        "/auth/user/email",
        content="invalid json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
