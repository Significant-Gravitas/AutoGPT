from datetime import UTC, datetime
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from prisma.enums import AuthAccessBlockKind

from backend.api.rest_api import app as rest_app
from backend.data.access_blocks import AccessBlock
from backend.util.exceptions import NotFoundError

from .access_block_admin_routes import router as access_block_router
from .user_admin_routes import router as user_admin_router

# Bad input reaches the client as the app-wide handlers map it (400, 404).
app = fastapi.FastAPI(exception_handlers=rest_app.exception_handlers)
app.include_router(access_block_router)
app.include_router(user_admin_router)

client = fastapi.testclient.TestClient(app)

_MOCK_MODULE = "backend.api.features.admin.access_block_admin_routes"
_BLOCK_ID = "6a1b7c3d-0000-4000-8000-000000000002"

_ADMIN_ROUTES = [
    ("GET", "/admin/access-blocks"),
    ("POST", "/admin/access-blocks"),
    ("DELETE", f"/admin/access-blocks/{_BLOCK_ID}"),
    ("GET", "/admin/users/5d0a1c2e-0000-4000-8000-000000000001/ip-report"),
]


@pytest.fixture(autouse=True)
def setup_app_admin_auth(mock_jwt_admin):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_admin["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


@pytest.mark.parametrize("method, path", _ADMIN_ROUTES)
def test_every_route_refuses_a_non_admin(
    method: str, path: str, mock_jwt_user, mocker: pytest_mock.MockerFixture
) -> None:
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    data_calls = [
        mocker.patch(f"{_MOCK_MODULE}.{name}", new_callable=AsyncMock)
        for name in (
            "list_active_access_blocks",
            "create_access_block",
            "delete_access_block",
        )
    ] + [
        mocker.patch(
            "backend.api.features.admin.user_admin_routes.get_account_ip_report",
            new_callable=AsyncMock,
        )
    ]

    response = client.request(
        method,
        path,
        json={"kind": "IP", "value": "203.0.113.7", "reason": "abuse"},
    )

    assert response.status_code == 403
    for call in data_calls:
        call.assert_not_awaited()


def test_create_records_the_admin_who_blocked(
    admin_user_id: str, mocker: pytest_mock.MockerFixture
) -> None:
    block = AccessBlock(
        id=_BLOCK_ID,
        kind=AuthAccessBlockKind.IP,
        value="203.0.113.7",
        reason="sign-up farm",
        created_by_id=admin_user_id,
        created_by_email="test-admin@example.com",
        created_at=datetime(2026, 10, 21, tzinfo=UTC),
        expires_at=None,
    )
    create = mocker.patch(
        f"{_MOCK_MODULE}.create_access_block",
        new_callable=AsyncMock,
        return_value=block,
    )

    response = client.post(
        "/admin/access-blocks",
        json={
            "kind": "IP",
            "value": "203.0.113.7",
            "reason": "sign-up farm",
            "expires_at": "2099-01-01T00:00:00Z",
        },
    )

    assert response.status_code == 201
    assert response.json()["id"] == _BLOCK_ID
    create.assert_awaited_once_with(
        kind=AuthAccessBlockKind.IP,
        value="203.0.113.7",
        reason="sign-up farm",
        expires_at=datetime(2099, 1, 1, tzinfo=UTC),
        created_by_id=admin_user_id,
    )


def test_an_expiry_without_a_time_zone_is_a_422(
    mocker: pytest_mock.MockerFixture,
) -> None:
    create = mocker.patch(f"{_MOCK_MODULE}.create_access_block", new_callable=AsyncMock)

    response = client.post(
        "/admin/access-blocks",
        json={
            "kind": "IP",
            "value": "203.0.113.7",
            "reason": "abuse",
            "expires_at": "2099-01-01T00:00:00",
        },
    )

    assert response.status_code == 422
    create.assert_not_awaited()


def test_blocking_a_public_email_provider_is_a_400() -> None:
    response = client.post(
        "/admin/access-blocks",
        json={"kind": "EMAIL_DOMAIN", "value": "gmail.com", "reason": "abuse"},
    )

    assert response.status_code == 400
    assert "public email provider" in response.json()["detail"]


def test_deleting_an_unknown_block_is_a_404(mocker: pytest_mock.MockerFixture) -> None:
    mocker.patch(
        f"{_MOCK_MODULE}.delete_access_block",
        new_callable=AsyncMock,
        side_effect=NotFoundError("Access block not found"),
    )

    response = client.delete(f"/admin/access-blocks/{_BLOCK_ID}")

    assert response.status_code == 404
