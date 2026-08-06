"""Authorization tests for the org-scoped credits routes in v1.py.

SECRT-2449: the credits balance/transaction/invoice/top-up routes resolve
their credit model through the request's org context, so for a real (pooled)
org they read/mutate the shared ``OrgBalance``. They must therefore require
org-level ``MANAGE_BILLING`` (owner or billing_manager) — a plain org member
must be rejected with 403. Personal-org owners always carry ``is_org_owner``,
so the gate is a no-op for them.

The gate is applied as an independent per-route dependency (there is no
shared router-level enforcement), so every gated route is asserted here:
dropping the dependency from any single route must fail this suite. Route
coverage is not left to the hand-maintained ``GATED_ROUTES`` list either —
``test_every_credits_route_is_gated_or_explicitly_exempt`` introspects the
mounted app, so a *newly added* ungated ``/credits`` route also fails.
"""

import inspect
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, Mock

import fastapi
import fastapi.testclient
import pydantic
import pytest
import pytest_mock
from autogpt_libs.auth.dependencies import get_request_context
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from autogpt_libs.auth.models import RequestContext
from autogpt_libs.auth.permissions import OrgAction
from fastapi.dependencies.models import Dependant
from fastapi.routing import APIRoute

from backend.data.model import AutoTopUpConfig, TransactionHistory

from .v1 import v1_router

app = fastapi.FastAPI()
app.include_router(v1_router)
client = fastapi.testclient.TestClient(app)

ORG_ID = "test-org"


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
#
# MANAGE_BILLING is granted to {owner, billing_manager} only (see
# autogpt_libs.auth.permissions._ORG_PERMISSIONS). ``org_admin`` is the
# important negative case: admins can rename the org and manage members but
# are deliberately excluded from billing, so widening the grant to admins must
# break this suite.
#
# Personal orgs are covered separately by the tests at the bottom of this
# module, which drive the *real* ``get_request_context`` resolution instead of
# overriding it — a role-flag row here could not tell a personal org apart
# from a team org, since the gate is a pure function of the role flags.
ROLE_CASES: dict[str, tuple[dict, int]] = {
    "org_owner": ({"owner": True, "admin": True}, 200),
    "billing_manager": ({"billing": True}, 200),
    "org_admin": ({"admin": True}, 403),
    "plain_member": ({}, 403),
}


class GatedRoute(pydantic.BaseModel):
    """One MANAGE_BILLING-gated endpoint and how to exercise it."""

    name: str
    method: str
    path: str
    # Predicate on the success response, to prove the route body really ran.
    check_ok: Callable[[Any], bool]
    body: dict | None = None
    # Attribute on ``CreditStubs`` for the first data-layer call the route body
    # makes. On a 403 it must never have been awaited — proving the gate
    # rejects during dependency resolution, before any billing data is touched.
    guard: str = "get_credit_model"


# Every route in v1.py carrying
# ``Security(requires_org_permission(OrgAction.MANAGE_BILLING))`` (via the
# ``BillingManagerContext`` alias). Kept in sync with the app by
# ``test_every_credits_route_is_gated_or_explicitly_exempt``; ``name`` is the
# endpoint function name, which is also the FastAPI route name.
GATED_ROUTES: list[GatedRoute] = [
    GatedRoute(
        name="get_user_credits",
        method="GET",
        path="/credits",
        check_ok=lambda r: r.json() == {"credits": 1000},
    ),
    GatedRoute(
        name="request_top_up",
        method="POST",
        path="/credits",
        body={"credit_amount": 500},
        check_ok=lambda r: r.json()["checkout_url"].startswith("https://"),
    ),
    GatedRoute(
        name="refund_top_up",
        method="POST",
        path="/credits/test-transaction-key/refund",
        body={"reason": "duplicate charge"},
        check_ok=lambda r: r.json() == 500,
    ),
    GatedRoute(
        name="fulfill_checkout",
        method="PATCH",
        path="/credits",
        check_ok=lambda r: r.content == b"",
    ),
    GatedRoute(
        name="configure_user_auto_top_up",
        method="POST",
        path="/credits/auto-top-up",
        body={"amount": 500, "threshold": 100},
        check_ok=lambda r: r.json() == "Auto top-up settings updated",
    ),
    GatedRoute(
        name="get_user_auto_top_up",
        method="GET",
        path="/credits/auto-top-up",
        check_ok=lambda r: r.json() == {"amount": 500, "threshold": 100},
        # This route reads the config helper directly, not the org credit model.
        guard="get_auto_top_up",
    ),
    GatedRoute(
        name="manage_payment_method",
        method="GET",
        path="/credits/manage",
        check_ok=lambda r: r.json() == {"url": "https://billing.example.com/portal"},
    ),
    GatedRoute(
        name="get_credit_history",
        method="GET",
        path="/credits/transactions",
        check_ok=lambda r: r.json()["transactions"] == [],
    ),
    GatedRoute(
        name="get_refund_requests",
        method="GET",
        path="/credits/refunds",
        check_ok=lambda r: r.json() == [],
    ),
    GatedRoute(
        name="list_invoices",
        method="GET",
        path="/credits/invoices",
        check_ok=lambda r: r.json() == [],
    ),
]


# ``/credits*`` routes that are deliberately NOT behind MANAGE_BILLING, keyed
# by endpoint function name with the reason they are exempt. Adding an entry
# here is an explicit product decision, not a way to silence the introspection
# test below.
UNGATED_CREDITS_ROUTES: dict[str, str] = {
    "get_subscription_status": "subscriptions are user-level, not org-pooled",
    "update_subscription_tier": "subscriptions are user-level, not org-pooled",
    "stripe_webhook": "unauthenticated by design; verified by Stripe signature",
}


class CreditStubs(pydantic.BaseModel):
    """Patched data-layer entry points reachable from the gated routes."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    get_credit_model: AsyncMock
    get_auto_top_up: AsyncMock
    set_auto_top_up: AsyncMock
    model: Mock


@pytest.fixture(autouse=True)
def _auth(mock_jwt_user):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def credit_stubs(mocker: pytest_mock.MockFixture) -> CreditStubs:
    """Patch every data-layer call the gated routes make.

    ``get_credit_model``/``get_auto_top_up``/``set_auto_top_up`` are async
    functions, so ``mocker.patch`` yields AsyncMocks and
    ``await get_credit_model(...)`` resolves to ``model``.
    """
    model = Mock()
    model.get_credits = AsyncMock(return_value=1000)
    model.get_transaction_history = AsyncMock(
        return_value=TransactionHistory(transactions=[], next_transaction_time=None)
    )
    model.list_invoices = AsyncMock(return_value=[])
    model.top_up_intent = AsyncMock(return_value="https://checkout.example.com/s")
    model.top_up_refund = AsyncMock(return_value=500)
    model.top_up_credits = AsyncMock(return_value=None)
    model.fulfill_checkout = AsyncMock(return_value=None)
    model.create_billing_portal_session = AsyncMock(
        return_value="https://billing.example.com/portal"
    )
    model.get_refund_requests = AsyncMock(return_value=[])
    return CreditStubs(
        get_credit_model=mocker.patch(
            "backend.api.features.v1.get_credit_model", return_value=model
        ),
        get_auto_top_up=mocker.patch(
            "backend.api.features.v1.get_auto_top_up",
            return_value=AutoTopUpConfig(amount=500, threshold=100),
        ),
        set_auto_top_up=mocker.patch("backend.api.features.v1.set_auto_top_up"),
        model=model,
    )


def _use_role(role: str, user_id: str) -> int:
    flags, expected = ROLE_CASES[role]
    ctx = _ctx(user_id, **flags)

    async def _override() -> RequestContext:
        return ctx

    app.dependency_overrides[get_request_context] = _override
    return expected


@pytest.mark.parametrize("route", GATED_ROUTES, ids=lambda r: r.name)
@pytest.mark.parametrize("role", list(ROLE_CASES))
def test_credits_route_requires_manage_billing(
    role: str, route: GatedRoute, test_user_id: str, credit_stubs: CreditStubs
):
    expected = _use_role(role, test_user_id)

    resp = client.request(route.method, route.path, json=route.body)

    assert resp.status_code == expected, resp.text
    if expected == 200:
        assert route.check_ok(resp), resp.text
        if route.guard == "get_credit_model":
            # The org the gate resolved is what the credit model is scoped to,
            # so a wrong-org regression fails here on every gated route — not
            # only in the single personal-org resolution test below.
            assert credit_stubs.get_credit_model.await_args.args[1] == ORG_ID
    else:
        assert resp.json()["detail"] == "Missing org permission: MANAGE_BILLING"
        # The gate rejects during dependency resolution, before the route body
        # ever reaches the org-pooled balance.
        getattr(credit_stubs, route.guard).assert_not_awaited()


def _enforced_org_actions(dependant: Dependant) -> set[OrgAction]:
    """Org actions enforced by a route's dependency tree.

    ``requires_org_permission(*actions)`` returns a closure, so the actions it
    enforces are read back off the closure rather than re-derived from the
    route signature.
    """
    enforced: set[OrgAction] = set()
    for sub in dependant.dependencies:
        call = sub.call
        if (
            inspect.isfunction(call)
            and call.__qualname__ == "requires_org_permission.<locals>._dependency"
        ):
            enforced.update(inspect.getclosurevars(call).nonlocals["actions"])
        enforced |= _enforced_org_actions(sub)
    return enforced


def test_every_credits_route_is_gated_or_explicitly_exempt():
    """Introspect the mounted app so a *new* ungated /credits route fails.

    ``GATED_ROUTES`` above is hand-maintained, so on its own it can only prove
    that the routes someone remembered to list are gated. This walks the real
    routing table instead: every ``/credits*`` route must either carry the
    MANAGE_BILLING dependency or be an explicit, documented exemption.
    """
    gated: set[str] = set()
    ungated: set[str] = set()
    for route in app.routes:
        if not isinstance(route, APIRoute) or not route.path.startswith("/credits"):
            continue
        if OrgAction.MANAGE_BILLING in _enforced_org_actions(route.dependant):
            gated.add(route.name)
        else:
            ungated.add(route.name)

    assert gated, "no /credits routes found — did the router or prefix change?"
    assert ungated == set(UNGATED_CREDITS_ROUTES), (
        "A /credits route is not behind MANAGE_BILLING. Gate it with "
        "`ctx: BillingManagerContext`, or — if it is genuinely not org-pooled "
        "— add it to UNGATED_CREDITS_ROUTES with the reason. Unexpected: "
        f"{sorted(ungated - set(UNGATED_CREDITS_ROUTES))}"
    )
    assert gated == {route.name for route in GATED_ROUTES}, (
        "GATED_ROUTES is out of sync with the routes actually carrying the "
        "MANAGE_BILLING dependency; add the new route to GATED_ROUTES so the "
        "role matrix exercises it: "
        f"{sorted(gated ^ {route.name for route in GATED_ROUTES})}"
    )


def _org_member(
    *, owner: bool = False, admin: bool = False, billing: bool = False
) -> Mock:
    """A prisma ``OrgMember`` row as ``get_request_context`` expects it."""
    row = Mock()
    row.status = "ACTIVE"
    row.isOwner = owner
    row.isAdmin = admin
    row.isBillingManager = billing
    row.Org = Mock(deletedAt=None)
    return row


@pytest.fixture
def mock_prisma(mocker: pytest_mock.MockFixture) -> Mock:
    """Stub the prisma client that the real ``get_request_context`` uses."""
    prisma = mocker.patch("backend.data.db.prisma")
    prisma.orgmember.find_first = AsyncMock(return_value=None)
    prisma.orgmember.find_unique = AsyncMock(return_value=None)
    return prisma


def test_personal_org_owner_passes_real_context_resolution(
    mock_prisma: Mock, credit_stubs: CreditStubs
):
    """A personal-org user hits the real resolution path and is let through.

    No ``X-Org-Id`` header is sent, so ``get_request_context`` falls back to the
    user's personal org, whose membership row is always ``isOwner=True``. This
    deliberately does *not* override ``get_request_context``: the claim being
    tested is that the personal-org fallback yields owner rights, which a
    hand-built RequestContext could not prove.
    """
    mock_prisma.orgmember.find_first = AsyncMock(return_value=Mock(orgId="personal-1"))
    mock_prisma.orgmember.find_unique = AsyncMock(return_value=_org_member(owner=True))

    resp = client.get("/credits")

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"credits": 1000}
    # The personal-org fallback ran (no X-Org-Id header was supplied) and its
    # org id is what the credit model was resolved against.
    mock_prisma.orgmember.find_first.assert_awaited_once()
    assert credit_stubs.get_credit_model.await_args.args[1] == "personal-1"


def test_plain_member_of_shared_org_rejected_real_context_resolution(
    mock_prisma: Mock, credit_stubs: CreditStubs
):
    """The same real resolution path rejects a plain member of a pooled org."""
    mock_prisma.orgmember.find_unique = AsyncMock(return_value=_org_member())

    resp = client.get("/credits", headers={"X-Org-Id": "shared-org"})

    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"] == "Missing org permission: MANAGE_BILLING"
    credit_stubs.get_credit_model.assert_not_awaited()
