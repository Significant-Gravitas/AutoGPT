"""The admin routes behind a sales-led team plan: who may call them and what
they accept. ``org_plan_e2e_test.py`` drives them against the database."""

from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from autogpt_libs.auth.jwt_utils import get_jwt_payload

from backend.api.utils.exceptions import add_exception_handlers
from backend.copilot.rate_limit import SubscriptionTier
from backend.data.org_plan_admin import OrgCreditGrantResult

from . import org_plan_admin_routes
from .org_plan_admin_routes import router

app = fastapi.FastAPI()
app.include_router(router)
add_exception_handlers(app)
client = fastapi.testclient.TestClient(app)

_ROUTES = "backend.api.features.admin.org_plan_admin_routes"


@pytest.fixture(autouse=True)
def as_admin(mock_jwt_admin):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_admin["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def test_only_an_admin_may_call_them(mock_jwt_user, mocker: pytest_mock.MockerFixture):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    write = mocker.patch(f"{_ROUTES}.set_org_plan", AsyncMock())

    response = client.put(
        "/admin/org-plans/orgs/org-1", json={"tier": "PRO", "seats": 2}
    )

    assert response.status_code == 403
    write.assert_not_awaited()


@pytest.mark.parametrize("tier", ["NO_TIER", "TRIAL"])
def test_a_plan_gives_a_paid_tier(tier: str, mocker: pytest_mock.MockerFixture):
    write = mocker.patch(f"{_ROUTES}.set_org_plan", AsyncMock())

    response = client.put(
        "/admin/org-plans/orgs/org-1", json={"tier": tier, "seats": 2}
    )

    assert response.status_code == 422
    write.assert_not_awaited()


def test_a_plan_needs_at_least_one_seat(mocker: pytest_mock.MockerFixture):
    write = mocker.patch(f"{_ROUTES}.set_org_plan", AsyncMock())
    response = client.put(
        "/admin/org-plans/orgs/org-1", json={"tier": "BUSINESS", "seats": 0}
    )
    assert response.status_code == 422
    write.assert_not_awaited()


def test_setting_a_plan_passes_the_admin_and_the_plan(
    mocker: pytest_mock.MockerFixture, admin_user_id: str
):
    write = mocker.patch(f"{_ROUTES}.set_org_plan", AsyncMock())
    write.return_value = {
        "org_id": "org-1",
        "name": "Acme",
        "slug": "acme",
        "owner_user_id": None,
        "owner_email": None,
        "plan": None,
        "members": 1,
        "pending_invitations": 0,
        "seats_in_use": 1,
        "balance": 0,
    }

    response = client.put(
        "/admin/org-plans/orgs/org-1",
        json={"tier": "BUSINESS", "seats": 5, "note": "INV-7"},
    )

    assert response.status_code == 200, response.text
    org_id, actor, plan = write.await_args.args
    assert (org_id, actor) == ("org-1", admin_user_id)
    assert (plan.tier, plan.seats, plan.status, plan.note) == (
        SubscriptionTier.BUSINESS,
        5,
        "active",
        "INV-7",
    )


def test_credits_carry_their_reference(
    mocker: pytest_mock.MockerFixture, admin_user_id: str
):
    grant = mocker.patch(
        f"{_ROUTES}.grant_org_credits",
        AsyncMock(
            return_value=OrgCreditGrantResult(
                org_id="org-1", reference="INV-7", granted=True, balance=500
            )
        ),
    )

    response = client.post(
        "/admin/org-plans/orgs/org-1/credits",
        json={"amount": 500, "reference": "INV-7"},
    )

    assert response.status_code == 200
    grant.assert_awaited_once_with("org-1", admin_user_id, 500, "INV-7", None)


@pytest.mark.parametrize(
    "body", [{"amount": 0, "reference": "INV-7"}, {"amount": 5, "reference": ""}]
)
def test_credits_need_an_amount_and_a_reference(
    body: dict, mocker: pytest_mock.MockerFixture
):
    grant = mocker.patch(f"{_ROUTES}.grant_org_credits", AsyncMock())
    response = client.post("/admin/org-plans/orgs/org-1/credits", json=body)
    assert response.status_code == 422
    grant.assert_not_awaited()


def test_finding_orgs_by_an_unknown_email_is_a_404(
    mocker: pytest_mock.MockerFixture,
):
    mocker.patch.object(
        org_plan_admin_routes, "find_user_id_by_email", AsyncMock(return_value=None)
    )
    response = client.get("/admin/org-plans/orgs", params={"email": "x@example.com"})
    assert response.status_code == 404


def test_finding_orgs_needs_an_account():
    assert client.get("/admin/org-plans/orgs").status_code == 422
