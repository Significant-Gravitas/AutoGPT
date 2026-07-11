"""Tests for the org-scoped per-team spend breakdown route.

SECRT-2450: ``GET /orgs/{org_id}/spend`` aggregates the org's ``USAGE`` (debit)
ledger by the team each debit was attributed to. It must require org-level
``MANAGE_BILLING`` (owner or billing_manager); a plain member is rejected with
403. Usage with no team attribution (org-home / legacy migrations) forms a
single NULL-team bucket, top-ups are never counted as spend, and
``total_spent`` is reported as a positive magnitude of credits spent.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from autogpt_libs.auth.dependencies import get_request_context
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from autogpt_libs.auth.models import RequestContext
from prisma.enums import CreditTransactionType

from .routes import router

ORG_ID = "test-org"

app = fastapi.FastAPI()
app.include_router(router, prefix="/orgs")
client = fastapi.testclient.TestClient(app)


def _ctx(
    user_id: str,
    *,
    owner: bool = False,
    admin: bool = False,
    billing: bool = False,
    org_id: str = ORG_ID,
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
# MANAGE_BILLING is owner or billing_manager only; a plain member is rejected.
ROLE_CASES: dict[str, tuple[dict, int]] = {
    "org_owner": (dict(owner=True, admin=True), 200),
    "billing_manager": (dict(billing=True), 200),
    "plain_member": (dict(), 403),
}


@pytest.fixture(autouse=True)
def _auth(mock_jwt_user):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def _team(team_id: str, name: str) -> SimpleNamespace:
    # ``name`` is a reserved Mock kwarg, so use a plain namespace for the
    # id/name attributes the aggregation reads.
    return SimpleNamespace(id=team_id, name=name)


@pytest.fixture
def spend_prisma(mocker: pytest_mock.MockFixture):
    """Patch the prisma client used by the spend aggregation.

    ``group_by`` returns two attributed team buckets plus a NULL-team bucket,
    each with a negative USAGE ``_sum`` (a debit); ``find_many`` resolves the
    two team names.
    """
    prisma = MagicMock()
    prisma.orgcredittransaction.group_by = AsyncMock(
        return_value=[
            {"teamId": "team-a", "_sum": {"amount": -300}, "_count": {"_all": 3}},
            {"teamId": "team-b", "_sum": {"amount": -150}, "_count": {"_all": 2}},
            {"teamId": None, "_sum": {"amount": -50}, "_count": {"_all": 1}},
        ]
    )
    prisma.team.find_many = AsyncMock(
        return_value=[_team("team-a", "Alpha"), _team("team-b", "Bravo")]
    )
    mocker.patch("backend.data.org_credit.prisma", prisma)
    return prisma


def _use_role(role: str, user_id: str) -> int:
    flags, expected = ROLE_CASES[role]
    ctx = _ctx(user_id, **flags)

    async def _override() -> RequestContext:
        return ctx

    app.dependency_overrides[get_request_context] = _override
    return expected


@pytest.mark.parametrize("role", list(ROLE_CASES))
def test_spend_requires_manage_billing(role, test_user_id, spend_prisma):
    expected = _use_role(role, test_user_id)

    resp = client.get(f"/orgs/{ORG_ID}/spend")

    assert resp.status_code == expected
    if expected == 403:
        # The gate rejects during dependency resolution, before the route body
        # ever runs the aggregation query.
        spend_prisma.orgcredittransaction.group_by.assert_not_awaited()


def test_spend_aggregation_shape(test_user_id, spend_prisma):
    _use_role("org_owner", test_user_id)

    resp = client.get(f"/orgs/{ORG_ID}/spend")

    assert resp.status_code == 200
    # Highest spend first; USAGE debit sums negated into positive magnitudes;
    # the NULL-team bucket is preserved with team_name null.
    assert resp.json()["teams"] == [
        {
            "team_id": "team-a",
            "team_name": "Alpha",
            "total_spent": 300,
            "transaction_count": 3,
        },
        {
            "team_id": "team-b",
            "team_name": "Bravo",
            "total_spent": 150,
            "transaction_count": 2,
        },
        {
            "team_id": None,
            "team_name": None,
            "total_spent": 50,
            "transaction_count": 1,
        },
    ]


def test_spend_empty_history_returns_empty_list(test_user_id, spend_prisma):
    spend_prisma.orgcredittransaction.group_by = AsyncMock(return_value=[])
    _use_role("billing_manager", test_user_id)

    resp = client.get(f"/orgs/{ORG_ID}/spend")

    assert resp.status_code == 200
    assert resp.json() == {"teams": []}
    # No team ids to resolve -> no second query over the Team table.
    spend_prisma.team.find_many.assert_not_awaited()


def test_spend_time_window_passed_to_query(test_user_id, spend_prisma):
    _use_role("org_owner", test_user_id)

    resp = client.get(
        f"/orgs/{ORG_ID}/spend",
        params={"from": "2026-01-01T00:00:00Z", "to": "2026-02-01T00:00:00Z"},
    )

    assert resp.status_code == 200
    where = spend_prisma.orgcredittransaction.group_by.call_args.kwargs["where"]
    assert where["createdAt"] == {
        "gte": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "lte": datetime(2026, 2, 1, tzinfo=timezone.utc),
    }
    # Only active USAGE debits for this org are aggregated — top-ups are never
    # counted as spend.
    assert where["type"] == CreditTransactionType.USAGE
    assert where["orgId"] == ORG_ID
    assert where["isActive"] is True
