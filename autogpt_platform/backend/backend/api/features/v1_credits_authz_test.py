"""Authorization tests for the org-scoped credits routes.

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
every test in this module drives the real mounted application, so a
``/api/credits`` route served by any module is in view, and two introspection
tests walk the routing table.

Every ungated route is asserted by the property that makes it safe rather
than listed by name (SECRT-2650): a name in an exemption list survives the
removal of the thing that justified it.
"""

import ast
import importlib.util
import inspect
import sys
import textwrap
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

from backend.api.rest_api import app
from backend.data.credit import UserCreditBase, get_credit_model
from backend.data.model import AutoTopUpConfig, TransactionHistory
from backend.data.org_credit import OrgCreditModel

# The real application, not a locally-mounted subset: a private app carries only
# the routers someone remembered to include, so a /credits route that moves to
# another module leaves the suite passing over nothing.
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


# Every route carrying
# ``Security(requires_org_permission(OrgAction.MANAGE_BILLING))`` (via the
# ``BillingManagerContext`` alias). Kept in sync with the app by
# ``test_every_credits_route_is_gated_or_explicitly_exempt``; ``name`` is the
# endpoint function name, which is also the FastAPI route name.
GATED_ROUTES: list[GatedRoute] = [
    GatedRoute(
        name="get_user_credits",
        method="GET",
        path="/api/credits",
        check_ok=lambda r: r.json() == {"credits": 1000},
    ),
    GatedRoute(
        name="request_top_up",
        method="POST",
        path="/api/credits",
        body={"credit_amount": 500},
        check_ok=lambda r: r.json()["checkout_url"].startswith("https://"),
    ),
    GatedRoute(
        name="refund_top_up",
        method="POST",
        path="/api/credits/test-transaction-key/refund",
        body={"reason": "duplicate charge"},
        check_ok=lambda r: r.json() == 500,
    ),
    GatedRoute(
        name="fulfill_checkout",
        method="PATCH",
        path="/api/credits",
        check_ok=lambda r: r.content == b"",
    ),
    GatedRoute(
        name="configure_user_auto_top_up",
        method="POST",
        path="/api/credits/auto-top-up",
        body={"amount": 500, "threshold": 100},
        check_ok=lambda r: r.json() == "Auto top-up settings updated",
    ),
    GatedRoute(
        name="get_credit_history",
        method="GET",
        path="/api/credits/transactions",
        check_ok=lambda r: r.json()["transactions"] == [],
    ),
    GatedRoute(
        name="get_refund_requests",
        method="GET",
        path="/api/credits/refunds",
        check_ok=lambda r: r.json() == [],
    ),
    GatedRoute(
        name="list_invoices",
        method="GET",
        path="/api/credits/invoices",
        check_ok=lambda r: r.json() == [],
    ),
]


class CreditStubs(pydantic.BaseModel):
    """Patched data-layer entry points reachable from the gated routes."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    get_credit_model: AsyncMock
    get_auto_top_up: AsyncMock
    set_auto_top_up: AsyncMock
    model: Mock


@pytest.fixture(autouse=True)
def _auth(mock_jwt_user):
    # Restored, not cleared: this is the real app, and the session-scoped
    # SpinTestServer fixture keeps its own get_user_id override on it.
    previous = dict(app.dependency_overrides)
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()
    app.dependency_overrides.update(previous)


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
    # manage_payment_method moved to the subscriptions module with #14477 and
    # resolves that module's own get_credit_model — so the suite spans two
    # modules and both bindings need stubbing.
    mocker.patch(
        "backend.api.features.billing.subscriptions.routes.get_credit_model",
        return_value=model,
    )
    return CreditStubs(
        get_credit_model=mocker.patch(
            "backend.api.features.billing.credits.routes.get_credit_model",
            return_value=model,
        ),
        get_auto_top_up=mocker.patch(
            "backend.api.features.billing.credits.routes.get_auto_top_up",
            return_value=AutoTopUpConfig(amount=500, threshold=100),
        ),
        set_auto_top_up=mocker.patch(
            "backend.api.features.billing.credits.routes.set_auto_top_up"
        ),
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
        # The org the gate resolved is what the credit model is scoped to, so a
        # wrong-org regression fails here on every gated route — not only in the
        # single personal-org resolution test below.
        assert credit_stubs.get_credit_model.await_args.args[1] == ORG_ID
    else:
        assert resp.json()["detail"] == "Missing org permission: MANAGE_BILLING"
        # The gate rejects during dependency resolution, before the route body
        # ever reaches the org-pooled balance.
        credit_stubs.get_credit_model.assert_not_awaited()


@pytest.mark.parametrize(
    "path, expected_json",
    [
        ("/api/credits/auto-top-up", {"amount": 500, "threshold": 100}),
        ("/api/credits/manage", {"url": "https://billing.example.com/portal"}),
    ],
    ids=["get_user_auto_top_up", "manage_payment_method"],
)
def test_user_scoped_credits_route_allows_plain_member(
    path: str, expected_json: dict, test_user_id: str, credit_stubs: CreditStubs
):
    """The two GETs that serve the caller's own data must stay ungated.

    ``get_auto_top_up`` reads the caller's ``User.top_up_config`` and
    ``create_billing_portal_session`` is not overridden by ``OrgCreditModel``,
    so it mints a portal for the caller's own Stripe customer. Gating either
    would deny a plain member their own settings while protecting no org data.
    """
    _use_role("plain_member", test_user_id)

    resp = client.get(path)

    assert resp.status_code == 200, resp.text
    assert resp.json() == expected_json


def test_manage_payment_method_opens_the_callers_own_portal(
    test_user_id: str, credit_stubs: CreditStubs
):
    """Pin WHOSE portal the route opens, which the response cannot show.

    The exemption above rests on the portal being the caller's own. The stub
    returns the same URL whatever it is handed, so swapping the caller for the
    org id leaves the response identical and the assertion above green while
    changing which billing account is opened.
    """
    _use_role("plain_member", test_user_id)

    resp = client.get("/api/credits/manage")

    assert resp.status_code == 200, resp.text
    credit_stubs.model.create_billing_portal_session.assert_awaited_once_with(
        test_user_id
    )


@pytest.mark.parametrize(
    "route_name, call",
    [
        ("request_top_up", lambda m: m.top_up_intent("user-1", 500)),
        ("refund_top_up", lambda m: m.top_up_refund("user-1", "key", {})),
        ("fulfill_checkout", lambda m: m.fulfill_checkout(user_id="user-1")),
    ],
    ids=["request_top_up", "refund_top_up", "fulfill_checkout"],
)
async def test_gated_top_up_route_is_unimplemented_for_pooled_orgs(
    route_name: str, call: Callable[[OrgCreditModel], Any]
):
    """The 200 rows above run on a mocked model; a real pooled org cannot.

    ``OrgCreditModel`` has no Stripe top-up, so a billing manager who passes the
    gate still cannot top up a pooled org. Asserting it here keeps the mocked
    happy path above from reading as a working feature.
    """
    assert route_name in {route.name for route in GATED_ROUTES}
    with pytest.raises(NotImplementedError):
        await call(OrgCreditModel(ORG_ID))


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


# The ``/api/credits*`` routes deliberately NOT behind MANAGE_BILLING. T250.2's
# audit (SECRT-2650) found that none of them needs it, so each is asserted by
# WHAT makes it safe: a name in an exemption list survives the removal of the
# thing that justified it, and eight of these sat outside this suite's view from
# the day it shipped, because it mounted one router rather than the app.
ADMIN_CREDITS_ROUTES = {
    "add_user_credits",
    "admin_get_all_user_history",
    "export_copilot_weekly_usage",
    "export_credit_transactions",
}
TRIAL_CREDITS_ROUTES = {
    "cancel_trial",
    "confirm_trial",
    "get_trial_status",
    "start_trial_checkout",
}
# Routes that never resolve the org-pooled credit model at all. The reason is
# prose; the exemption is checked by the transitive walk at the bottom of this
# module.
POOLED_MODEL_UNREACHABLE_ROUTES: dict[str, str] = {
    "get_subscription_status": "subscriptions are user-level, not org-pooled",
    "update_subscription_tier": "subscriptions are user-level, not org-pooled",
    "get_user_auto_top_up": "reads the caller's own User.top_up_config, not org data",
    "stripe_webhook": (
        "unauthenticated by design, verified by Stripe signature; it fulfils "
        "against UserCredit() directly and resolves no org"
    ),
}


def _dependency_names(dependant: Dependant) -> set[str]:
    """Every dependency call name in a route's flattened tree."""
    names: set[str] = set()
    for sub_dep in dependant.dependencies:
        call = sub_dep.call
        if call is not None:
            names.add(getattr(call, "__name__", type(call).__name__))
        names |= _dependency_names(sub_dep)
    return names


def test_every_credits_route_is_gated_or_explicitly_exempt():
    """Introspect the routing table so a *new* ungated /credits route fails.

    ``GATED_ROUTES`` above is hand-maintained, so on its own it can only prove
    that the routes someone remembered to list are gated. This walks every
    ``/api/credits*`` route the real application serves — whichever module
    serves it — and makes each ungated one prove its exemption.
    """
    gated: set[str] = set()
    unexplained: set[str] = set()
    for route in app.routes:
        if not isinstance(route, APIRoute) or not route.path.startswith("/api/credits"):
            continue
        if OrgAction.MANAGE_BILLING in _enforced_org_actions(route.dependant):
            gated.add(route.name)
        elif route.name in ADMIN_CREDITS_ROUTES:
            # Safe because an admin JWT role claim gates it, which no org role
            # reaches — not even the org owner. Assert that, not the name.
            assert "requires_admin_user" in _dependency_names(route.dependant), (
                f"{route.name} is treated as admin-gated but no longer resolves "
                "requires_admin_user; it now needs MANAGE_BILLING or its own reason"
            )
        elif route.name in TRIAL_CREDITS_ROUTES:
            # Safe because it resolves no org context at all, so there is no
            # pooled balance for it to read.
            assert "get_request_context" not in _dependency_names(route.dependant), (
                f"{route.name} now resolves an org context; it can reach pooled "
                "credit and needs gating or its own documented reason"
            )
        elif route.name in POOLED_MODEL_UNREACHABLE_ROUTES:
            assert not _reaches_credit_model(route.endpoint), (
                f"{route.name} now reaches get_credit_model, so it can resolve a "
                "pooled org balance; gate it or re-argue the exemption"
            )
        elif route.name == "manage_payment_method":
            # Safe because OrgCreditModel inherits create_billing_portal_session
            # rather than overriding it, so the route mints a portal for the
            # caller's own Stripe customer even under a pooled org.
            assert (
                OrgCreditModel.create_billing_portal_session
                is UserCreditBase.create_billing_portal_session
            ), (
                "OrgCreditModel now overrides create_billing_portal_session, so "
                "manage_payment_method can act on the org's billing account and "
                "needs MANAGE_BILLING"
            )
        else:
            unexplained.add(route.name)

    assert gated, "no /api/credits routes found — did the router or prefix change?"
    assert not unexplained, (
        "A /api/credits route is neither behind MANAGE_BILLING nor covered by an "
        "asserted exemption. Gate it with `ctx: BillingManagerContext`, or — if "
        "it genuinely serves no org-pooled data — add it to the set above whose "
        f"assertion says why it is safe. Unexpected: {sorted(unexplained)}"
    )
    assert gated == {route.name for route in GATED_ROUTES}, (
        "GATED_ROUTES is out of sync with the routes actually carrying the "
        "MANAGE_BILLING dependency; add the new route to GATED_ROUTES so the "
        "role matrix exercises it: "
        f"{sorted(gated ^ {route.name for route in GATED_ROUTES})}"
    )


# Routes that resolve the org-pooled credit model without the gate, with the
# reason. Anything else reaching ``get_credit_model`` must be gated.
ORG_BALANCE_UNGATED: dict[str, str] = {
    "execute_graph": (
        "the executor spends the pooled balance; gating it would stop plain "
        "members running agents, and it reveals no balance (402 only)"
    ),
    "manage_payment_method": (
        "opens the caller's own Stripe portal; OrgCreditModel does not override "
        "create_billing_portal_session"
    ),
    "get_home_dashboard": (
        "SECRT-2648 withholds /home's pooled balance from a member without "
        "MANAGE_BILLING, but it gates the read inside the service, where "
        "_get_credits calls check_org_permission, and this walk reads the "
        "route's DEPENDENCY tree — so the route still resolves the credit model "
        "with no route-level gate and this entry stays; removing it turns this "
        "test red"
    ),
}

# How deep the walk below follows calls. ``/home`` reaches the credit model
# three hops down (get_home_dashboard -> build_home_dashboard ->
# _load_home_source_data -> _get_credits), so 4 leaves one hop of headroom;
# raising it to 6 adds no further routes.
_CREDIT_MODEL_MAX_HOPS = 4
_WALKED_MODULES = ("backend.api.", "backend.data.")


def test_every_org_balance_reader_is_gated_or_explicitly_exempt():
    """Catch org-pooled balance access outside the ``/credits`` prefix.

    The path-prefix test above cannot see ``execute_graph``, which already reads
    the pooled balance from ``/graphs/{id}/execute``, nor a future org-billing
    route under another prefix.

    Coverage is every route reaching ``get_credit_model`` within
    ``_CREDIT_MODEL_MAX_HOPS`` calls through functions of
    ``backend.api``/``backend.data``, named either at module level or by an
    import inside the calling function — not every caller in the app: a longer
    chain, a call through an instance attribute, or a dynamically resolved name
    stays out of view. The one-hop source match this replaced could not see
    ``/home`` at all, which is SECRT-2648.
    """
    ungated = {
        route.name
        for route in app.routes
        if isinstance(route, APIRoute)
        and _reaches_credit_model(route.endpoint)
        and OrgAction.MANAGE_BILLING not in _enforced_org_actions(route.dependant)
    }

    assert ungated == set(ORG_BALANCE_UNGATED), (
        "A route resolves the org-pooled credit model without MANAGE_BILLING. "
        "Gate it with `ctx: BillingManagerContext`, or — if it genuinely serves "
        "the caller's own data — add it to ORG_BALANCE_UNGATED with the reason. "
        f"Unexpected: {sorted(ungated - set(ORG_BALANCE_UNGATED))}"
    )


def _defers_its_credit_import():
    """Stand-in for a route that imports the credit model inside its body."""
    from backend.data.credit import get_credit_model as _model_getter

    return _model_getter


def _defers_an_unrelated_import():
    """The same shape, importing something that never reaches the org pool."""
    from backend.data.model import AutoTopUpConfig

    return AutoTopUpConfig


@pytest.mark.parametrize(
    "func, expected",
    [(_defers_its_credit_import, True), (_defers_an_unrelated_import, False)],
    ids=["deferred credit import", "deferred unrelated import"],
)
def test_walk_resolves_function_local_imports(func: Callable, expected: bool):
    """A deferred import must not hide the credit model from the walk.

    ``backend/api`` and ``backend/data`` carry 116 function-local ``import
    backend…`` statements, so a route that defers its credit import is an
    ordinary shape here, not a contrivance — and a name bound that way is
    absent from ``__globals__``.
    """
    assert _reaches_credit_model(func) is expected


def test_unloaded_watched_import_fails_closed():
    """A watched module the walk cannot follow must break it, not shrink it."""
    with pytest.raises(AssertionError, match="backend.data.not_a_real_module"):
        _import_module("backend.data.not_a_real_module", "backend.api.features")


def _reaches_credit_model(endpoint: Callable) -> bool:
    """Whether ``endpoint`` reaches ``get_credit_model`` within the hop budget.

    Breadth-first, so every function is explored at its shortest distance from
    the endpoint and the budget means what it says. Names are resolved through
    each function's own module globals and compared by identity, so an aliased
    import is followed while a look-alike like ``get_user_credit_model`` — which
    returns the user's own wallet, never the org's — is not a match.
    """
    frontier, seen = [endpoint], {id(endpoint)}
    for _ in range(_CREDIT_MODEL_MAX_HOPS + 1):
        next_frontier: list[Callable] = []
        for func in frontier:
            for obj in _globals_referenced_by(func):
                if obj is get_credit_model:
                    return True
                if (
                    callable(obj)
                    and (getattr(obj, "__module__", "") or "").startswith(
                        _WALKED_MODULES
                    )
                    and id(obj) not in seen
                ):
                    seen.add(id(obj))
                    next_frontier.append(obj)
        frontier = next_frontier
    return False


def _globals_referenced_by(func: Callable) -> list[Any]:
    """Objects named in ``func``'s source, resolved to objects.

    Deferred imports are common here — 116 function-local ``import backend…``
    statements under ``backend/api`` and ``backend/data`` — and a name bound by
    one is absent from ``__globals__``, so the import's own bindings are
    resolved alongside them. Missing them would make a route that defers its
    credit import invisible to an invariant whose whole job is to see it.
    """
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    except (OSError, TypeError, SyntaxError):
        return []
    names = {**getattr(func, "__globals__", {}), **_local_import_bindings(tree, func)}
    referenced: list[Any] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in names:
            referenced.append(names[node.id])
        elif isinstance(node, ast.Attribute):
            obj = _resolve_dotted(node, names)
            if obj is not None:
                referenced.append(obj)
    return referenced


def _local_import_bindings(tree: ast.AST, func: Callable) -> dict[str, Any]:
    """Names bound by ``import``/``from … import`` statements inside ``func``."""
    package = getattr(func, "__module__", "").rsplit(".", 1)[0]
    bindings: dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # `import a.b.c` binds `a`, and _resolve_dotted walks the rest;
                # `import a.b.c as x` binds the leaf module to `x`.
                name = alias.asname or alias.name.split(".")[0]
                module = _import_module(alias.name if alias.asname else name, package)
                if module is not None:
                    bindings[name] = module
        elif isinstance(node, ast.ImportFrom):
            module = _import_module("." * node.level + (node.module or ""), package)
            if module is None:
                continue
            for alias in node.names:
                target = getattr(module, alias.name, None)
                if target is not None:
                    bindings[alias.asname or alias.name] = target
    return bindings


def _import_module(name: str, package: str) -> Any:
    """Resolve an import target to an already-loaded module.

    Nothing is imported here, because importing would run a deferred import's
    side effects inside a test. A watched module that is not loaded is a hole in
    the walk rather than a route that reaches nothing, so it raises instead of
    resolving to ``None``: under-reporting is the one answer this invariant must
    never give quietly.
    """
    try:
        absolute = importlib.util.resolve_name(name, package)
    except (ImportError, ValueError):
        return None
    module = sys.modules.get(absolute)
    assert module is not None or not absolute.startswith(_WALKED_MODULES), (
        f"{absolute} is imported inside a walked function but is not loaded, so "
        "the credit-model walk cannot follow it and would under-report. Import "
        "it in this suite, or teach the walk to read its source."
    )
    return module


def _resolve_dotted(node: ast.Attribute, module_globals: dict) -> Any:
    """Resolve a dotted reference such as ``credit.get_credit_model``."""
    attrs: list[str] = []
    value: ast.expr = node
    while isinstance(value, ast.Attribute):
        attrs.append(value.attr)
        value = value.value
    if not isinstance(value, ast.Name) or value.id not in module_globals:
        return None
    obj = module_globals[value.id]
    for attr in reversed(attrs):
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return obj


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

    resp = client.get("/api/credits")

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

    resp = client.get("/api/credits", headers={"X-Org-Id": "shared-org"})

    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"] == "Missing org permission: MANAGE_BILLING"
    credit_stubs.get_credit_model.assert_not_awaited()
