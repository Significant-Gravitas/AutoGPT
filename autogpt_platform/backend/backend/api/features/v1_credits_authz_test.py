"""Authorization tests for the org-scoped credits routes in v1.py.

SECRT-2449: the credits balance/transaction/invoice/top-up routes resolve
their credit model through the request's org context, so for a real (pooled)
org they read/mutate the shared ``OrgBalance``. They must therefore require
org-level ``MANAGE_BILLING`` (owner or billing_manager) — a plain org member
must be rejected with 403. Personal-org owners always carry ``is_org_owner``,
so the gate is a no-op for them.
"""

from unittest.mock import AsyncMock, Mock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from autogpt_libs.auth.dependencies import get_request_context
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from autogpt_libs.auth.models import RequestContext

from backend.data.model import TransactionHistory

from .v1 import v1_router

app = fastapi.FastAPI()
app.include_router(v1_router)
client = fastapi.testclient.TestClient(app)


def _ctx(
    user_id: str,
    *,
    owner: bool = False,
    admin: bool = False,
    billing: bool = False,
    org_id: str = "test-org",
) -> RequestContext:
    return RequestContext(
        user_id=user_id,
        org_id=org_id,
        team_id=None,
        is_org_owner=owner,
        is_org_admin=admin,
        is_org_billing_manager=billing,
        is_team_admin=False,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )


# role -> (RequestContext role flags, expected HTTP status).
# A personal-org owner is the sole member and always holds is_org_owner=True,
# so it maps to the same passing outcome as a team-org owner.
ROLE_CASES: dict[str, tuple[dict, int]] = {
    "org_owner": (dict(owner=True, admin=True), 200),
    "billing_manager": (dict(billing=True), 200),
    "plain_member": (dict(), 403),
    "personal_org_owner": (dict(owner=True, admin=True, org_id="personal-org"), 200),
}


@pytest.fixture(autouse=True)
def _auth(mock_jwt_user):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def credit_model(mocker: pytest_mock.MockFixture):
    """Patch get_credit_model with a stub whose methods return valid payloads.

    ``get_credit_model`` is an async function, so ``mocker.patch`` yields an
    AsyncMock; ``await get_credit_model(...)`` resolves to ``model``.
    """
    model = Mock()
    model.get_credits = AsyncMock(return_value=1000)
    model.get_transaction_history = AsyncMock(
        return_value=TransactionHistory(transactions=[], next_transaction_time=None)
    )
    model.list_invoices = AsyncMock(return_value=[])
    model.top_up_intent = AsyncMock(return_value="https://checkout.example.com/s")
    patched = mocker.patch(
        "backend.api.features.v1.get_credit_model", return_value=model
    )
    return patched, model


def _use_role(role: str, user_id: str) -> int:
    flags, expected = ROLE_CASES[role]
    ctx = _ctx(user_id, **flags)

    async def _override() -> RequestContext:
        return ctx

    app.dependency_overrides[get_request_context] = _override
    return expected


@pytest.mark.parametrize("role", list(ROLE_CASES))
def test_get_user_credits_requires_manage_billing(role, test_user_id, credit_model):
    patched, _ = credit_model
    expected = _use_role(role, test_user_id)

    resp = client.get("/credits")

    assert resp.status_code == expected
    if expected == 200:
        assert resp.json() == {"credits": 1000}
    else:
        # The gate rejects during dependency resolution, before the route body
        # ever resolves the org credit model.
        patched.assert_not_awaited()


@pytest.mark.parametrize("role", list(ROLE_CASES))
def test_get_credit_history_requires_manage_billing(role, test_user_id, credit_model):
    patched, _ = credit_model
    expected = _use_role(role, test_user_id)

    resp = client.get("/credits/transactions")

    assert resp.status_code == expected
    if expected == 200:
        assert resp.json()["transactions"] == []
    else:
        patched.assert_not_awaited()


@pytest.mark.parametrize("role", list(ROLE_CASES))
def test_list_invoices_requires_manage_billing(role, test_user_id, credit_model):
    patched, _ = credit_model
    expected = _use_role(role, test_user_id)

    resp = client.get("/credits/invoices")

    assert resp.status_code == expected
    if expected == 200:
        assert resp.json() == []
    else:
        patched.assert_not_awaited()


@pytest.mark.parametrize("role", list(ROLE_CASES))
def test_request_top_up_requires_manage_billing(role, test_user_id, credit_model):
    patched, _ = credit_model
    expected = _use_role(role, test_user_id)

    resp = client.post("/credits", json={"credit_amount": 500})

    assert resp.status_code == expected
    if expected == 200:
        assert "checkout_url" in resp.json()
    else:
        patched.assert_not_awaited()
