from datetime import UTC, datetime
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from autogpt_libs.auth.jwt_utils import get_jwt_payload

from backend.api.features.orgs.model import OrgResponse
from backend.data.auth_ip_events import AccountIPReport, EmailDomainInfo

from .user_admin_routes import router as user_admin_router

app = fastapi.FastAPI()
app.include_router(user_admin_router)

client = fastapi.testclient.TestClient(app)

_MOCK_MODULE = "backend.api.features.admin.user_admin_routes"
_TARGET_USER_ID = "5d0a1c2e-0000-4000-8000-000000000001"


@pytest.fixture(autouse=True)
def setup_app_admin_auth(mock_jwt_admin):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_admin["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def test_get_user_organizations_returns_the_target_users_orgs(
    mocker: pytest_mock.MockerFixture,
) -> None:
    org = OrgResponse(
        id="org-1",
        name="Acme",
        slug="acme",
        avatar_url=None,
        description=None,
        is_personal=False,
        member_count=0,
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    list_user_orgs = mocker.patch(
        f"{_MOCK_MODULE}.org_db.list_user_orgs",
        new_callable=AsyncMock,
        return_value=[org],
    )

    response = client.get(f"/admin/users/{_TARGET_USER_ID}/organizations")

    assert response.status_code == 200
    assert [o["slug"] for o in response.json()] == ["acme"]
    list_user_orgs.assert_awaited_once_with(_TARGET_USER_ID)


def test_get_user_organizations_requires_admin_role(
    mock_jwt_user, mocker: pytest_mock.MockerFixture
) -> None:
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    list_user_orgs = mocker.patch(
        f"{_MOCK_MODULE}.org_db.list_user_orgs", new_callable=AsyncMock
    )

    response = client.get(f"/admin/users/{_TARGET_USER_ID}/organizations")

    assert response.status_code == 403
    list_user_orgs.assert_not_awaited()


def test_get_user_ip_report_passes_the_configured_retention(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(f"{_MOCK_MODULE}.settings.config.auth_ip_event_retention_days", 30)
    report = AccountIPReport(
        retention_days=30,
        ips=[],
        domain=EmailDomainInfo(
            value="example.com",
            is_public_provider=False,
            other_account_count=0,
            block=None,
        ),
        related=[],
        related_total=0,
    )
    get_report = mocker.patch(
        f"{_MOCK_MODULE}.get_account_ip_report",
        new_callable=AsyncMock,
        return_value=report,
    )

    response = client.get(f"/admin/users/{_TARGET_USER_ID}/ip-report")

    assert response.status_code == 200
    assert response.json()["domain"]["value"] == "example.com"
    get_report.assert_awaited_once_with(_TARGET_USER_ID, 30)
