from datetime import datetime, timezone
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import prisma.enums
import pytest
import pytest_mock
from autogpt_libs.auth.jwt_utils import get_jwt_payload

from backend.copilot.model import ChatSession
from backend.copilot.session_types import ChatSessionStartType
from backend.data.invited_user import (
    BulkInvitedUserRowResult,
    BulkInvitedUsersResult,
    InvitedUserRecord,
)
from backend.data.model import User

from .user_admin_routes import router as user_admin_router

app = fastapi.FastAPI()
app.include_router(user_admin_router)

client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_admin_auth(mock_jwt_admin):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_admin["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def _sample_invited_user() -> InvitedUserRecord:
    now = datetime.now(timezone.utc)
    return InvitedUserRecord(
        id="invite-1",
        email="invited@example.com",
        status=prisma.enums.InvitedUserStatus.INVITED,
        auth_user_id=None,
        name="Invited User",
        tally_understanding=None,
        tally_status=prisma.enums.TallyComputationStatus.PENDING,
        tally_computed_at=None,
        tally_error=None,
        created_at=now,
        updated_at=now,
    )


def _sample_bulk_invited_users_result() -> BulkInvitedUsersResult:
    return BulkInvitedUsersResult(
        created_count=1,
        skipped_count=1,
        error_count=0,
        results=[
            BulkInvitedUserRowResult(
                row_number=1,
                email="invited@example.com",
                name=None,
                status="CREATED",
                message="Invite created",
                invited_user=_sample_invited_user(),
            ),
            BulkInvitedUserRowResult(
                row_number=2,
                email="duplicate@example.com",
                name=None,
                status="SKIPPED",
                message="An invited user with this email already exists",
                invited_user=None,
            ),
        ],
    )


def _sample_user() -> User:
    now = datetime.now(timezone.utc)
    return User(
        id="user-1",
        email="copilot@example.com",
        name="Copilot User",
        timezone="Europe/Madrid",
        created_at=now,
        updated_at=now,
        stripe_customer_id=None,
        top_up_config=None,
    )


def test_get_invited_users(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.api.features.admin.user_admin_routes.list_invited_users",
        AsyncMock(return_value=([_sample_invited_user()], 1)),
    )

    response = client.get("/admin/invited-users")

    assert response.status_code == 200
    data = response.json()
    assert len(data["invited_users"]) == 1
    assert data["invited_users"][0]["email"] == "invited@example.com"
    assert data["invited_users"][0]["status"] == "INVITED"
    assert data["pagination"]["total_items"] == 1
    assert data["pagination"]["current_page"] == 1
    assert data["pagination"]["page_size"] == 50


def test_create_invited_user(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.api.features.admin.user_admin_routes.create_invited_user",
        AsyncMock(return_value=_sample_invited_user()),
    )

    response = client.post(
        "/admin/invited-users",
        json={"email": "invited@example.com", "name": "Invited User"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "invited@example.com"
    assert data["name"] == "Invited User"


def test_bulk_create_invited_users(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.api.features.admin.user_admin_routes.bulk_create_invited_users_from_file",
        AsyncMock(return_value=_sample_bulk_invited_users_result()),
    )

    response = client.post(
        "/admin/invited-users/bulk",
        files={
            "file": ("invites.txt", b"invited@example.com\nduplicate@example.com\n")
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["created_count"] == 1
    assert data["skipped_count"] == 1
    assert data["results"][0]["status"] == "CREATED"
    assert data["results"][1]["status"] == "SKIPPED"


def test_revoke_invited_user(
    mocker: pytest_mock.MockerFixture,
) -> None:
    revoked = _sample_invited_user().model_copy(
        update={"status": prisma.enums.InvitedUserStatus.REVOKED}
    )
    mocker.patch(
        "backend.api.features.admin.user_admin_routes.revoke_invited_user",
        AsyncMock(return_value=revoked),
    )

    response = client.post("/admin/invited-users/invite-1/revoke")

    assert response.status_code == 200
    assert response.json()["status"] == "REVOKED"


def test_retry_invited_user_tally(
    mocker: pytest_mock.MockerFixture,
) -> None:
    retried = _sample_invited_user().model_copy(
        update={"tally_status": prisma.enums.TallyComputationStatus.RUNNING}
    )
    mocker.patch(
        "backend.api.features.admin.user_admin_routes.retry_invited_user_tally",
        AsyncMock(return_value=retried),
    )

    response = client.post("/admin/invited-users/invite-1/retry-tally")

    assert response.status_code == 200
    assert response.json()["tally_status"] == "RUNNING"


def test_search_copilot_users(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.api.features.admin.user_admin_routes.search_users",
        AsyncMock(return_value=[_sample_user()]),
    )

    response = client.get("/admin/copilot/users", params={"search": "copilot"})

    assert response.status_code == 200
    data = response.json()
    assert len(data["users"]) == 1
    assert data["users"][0]["email"] == "copilot@example.com"
    assert data["users"][0]["timezone"] == "Europe/Madrid"


def test_trigger_copilot_session(
    mocker: pytest_mock.MockerFixture,
) -> None:
    session = ChatSession.new(
        "user-1",
        start_type=ChatSessionStartType.AUTOPILOT_CALLBACK,
    )
    trigger = mocker.patch(
        "backend.api.features.admin.user_admin_routes.trigger_autopilot_session_for_user",
        AsyncMock(return_value=session),
    )

    response = client.post(
        "/admin/copilot/trigger",
        json={
            "user_id": "user-1",
            "start_type": ChatSessionStartType.AUTOPILOT_CALLBACK.value,
        },
    )

    assert response.status_code == 200
    assert response.json()["session_id"] == session.session_id
    assert response.json()["start_type"] == "AUTOPILOT_CALLBACK"
    assert trigger.await_args is not None
    assert trigger.await_args.args[0] == "user-1"
    assert (
        trigger.await_args.kwargs["start_type"]
        == ChatSessionStartType.AUTOPILOT_CALLBACK
    )


def test_trigger_copilot_session_returns_not_found(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.api.features.admin.user_admin_routes.trigger_autopilot_session_for_user",
        AsyncMock(side_effect=LookupError("User not found with ID: missing-user")),
    )

    response = client.post(
        "/admin/copilot/trigger",
        json={
            "user_id": "missing-user",
            "start_type": ChatSessionStartType.AUTOPILOT_NIGHTLY.value,
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "User not found with ID: missing-user"
