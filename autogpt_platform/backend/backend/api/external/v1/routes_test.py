import asyncio
import threading
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import pytest
from prisma.enums import APIKeyPermission

import backend.api.external.v1.integrations as integrations_mod
import backend.api.external.v1.routes as routes_mod
import backend.blocks
from backend.api.external.middleware import require_auth
from backend.api.external.v1.routes import v1_router
from backend.copilot.capabilities.block_meta import is_graph_only_block
from backend.copilot.rate_limit import UserPaywalledError
from backend.data.auth.base import APIAuthorizationInfo
from backend.util.exceptions import InsufficientBalanceError

app = fastapi.FastAPI()
app.include_router(v1_router)
client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_auth(test_user_id, v1_counter):
    """Give each case an authenticated principal and a fresh quota counter."""

    async def fake_require_auth() -> APIAuthorizationInfo:
        return APIAuthorizationInfo(
            user_id=test_user_id,
            scopes=list(APIKeyPermission),
            type="api_key",
            created_at=datetime.now(timezone.utc),
        )

    app.dependency_overrides[require_auth] = fake_require_auth
    yield
    app.dependency_overrides.clear()


def _stub_block(
    *,
    block_id: str = "00000000-0000-0000-0000-000000000001",
    name: str = "TestBlock",
    disabled: bool = False,
    capture: dict | None = None,
):
    """Build a minimal block stub for get_block(...) replacement.

    Async-iterable execute() yields one (name, value) pair so the route's
    `async for` loop can iterate without touching real block logic. If
    ``capture`` is supplied, the stub stores the kwargs it was called with
    so tests can assert on them (e.g. ``execution_context``).
    """
    block = MagicMock()
    block.id = block_id
    block.name = name
    block.disabled = disabled

    async def _execute(_data, **kwargs):
        if capture is not None:
            capture.update(kwargs)
        yield "result", "ok"

    block.execute = _execute
    return block


@pytest.fixture(autouse=True)
def stub_user_lookup(monkeypatch: pytest.MonkeyPatch, test_user_id: str):
    """The direct-block-execute route fetches the user to build an
    ExecutionContext; stub it so block-route tests don't need a real DB."""
    user = MagicMock()
    user.id = test_user_id
    user.timezone = "UTC"
    monkeypatch.setattr(
        "backend.api.external.v1.routes.user_db.get_user_by_id",
        AsyncMock(return_value=user),
    )


def test_zero_balance_returns_402_on_paid_block(monkeypatch: pytest.MonkeyPatch):
    """Zero-credit user calling a paid block must be rejected before execution."""
    block = _stub_block(name="PaidBlock")
    monkeypatch.setattr("backend.blocks.get_block", lambda _: block)
    monkeypatch.setattr(
        "backend.executor.utils.block_usage_cost", lambda *_a, **_k: (5, {})
    )
    spend_mock = AsyncMock(
        side_effect=InsufficientBalanceError(
            user_id="test-user-id",
            message="No credits left.",
            balance=0,
            amount=5,
        )
    )
    monkeypatch.setattr(
        "backend.executor.utils.get_user_credit_model",
        AsyncMock(return_value=MagicMock(spend_credits=spend_mock)),
    )

    response = client.post(f"/blocks/{block.id}/execute", json={})

    assert response.status_code == 402, f"got {response.status_code}: {response.text}"
    spend_mock.assert_awaited_once()


def test_paid_block_charges_then_runs(
    monkeypatch: pytest.MonkeyPatch, test_user_id: str
):
    """Happy path for a paid static-cost block: charge first, then execute."""
    block = _stub_block(name="PaidBlock")
    monkeypatch.setattr("backend.blocks.get_block", lambda _: block)
    monkeypatch.setattr(
        "backend.executor.utils.block_usage_cost",
        lambda *_a, **_k: (3, {"matched": True}),
    )
    spend_mock = AsyncMock(return_value=97)
    monkeypatch.setattr(
        "backend.executor.utils.get_user_credit_model",
        AsyncMock(return_value=MagicMock(spend_credits=spend_mock)),
    )

    response = client.post(f"/blocks/{block.id}/execute", json={})

    assert response.status_code == 200, f"got {response.status_code}: {response.text}"
    assert response.json() == {"result": ["ok"]}
    spend_mock.assert_awaited_once()
    kwargs = spend_mock.await_args.kwargs
    assert kwargs["user_id"] == test_user_id
    assert kwargs["cost"] == 3
    assert kwargs["metadata"].block_id == block.id
    assert kwargs["metadata"].block == "PaidBlock"
    assert kwargs["metadata"].input == {"matched": True}
    assert kwargs["metadata"].reason == "Direct external block execution of PaidBlock"


def test_free_block_runs_without_charging(monkeypatch: pytest.MonkeyPatch):
    """A block with cost == 0 should execute and never call spend_credits."""
    block = _stub_block(name="FreeBlock")
    monkeypatch.setattr("backend.blocks.get_block", lambda _: block)
    monkeypatch.setattr(
        "backend.executor.utils.block_usage_cost", lambda *_a, **_k: (0, {})
    )
    spend_mock = AsyncMock()
    monkeypatch.setattr(
        "backend.executor.utils.get_user_credit_model",
        AsyncMock(return_value=MagicMock(spend_credits=spend_mock)),
    )

    response = client.post(f"/blocks/{block.id}/execute", json={})

    assert response.status_code == 200, f"got {response.status_code}: {response.text}"
    assert response.json() == {"result": ["ok"]}
    spend_mock.assert_not_awaited()


def test_execute_block_forwards_execution_context(
    monkeypatch: pytest.MonkeyPatch, test_user_id: str
):
    """Regression for #12648: blocks that read ``execution_context`` (e.g.
    time blocks) crashed because this route didn't forward one. The route
    must build an ExecutionContext carrying the caller's user_id +
    timezone and pass it through to ``Block.execute``."""
    captured: dict = {}
    block = _stub_block(name="FreeBlock", capture=captured)
    monkeypatch.setattr("backend.blocks.get_block", lambda _: block)
    monkeypatch.setattr(
        "backend.executor.utils.block_usage_cost", lambda *_a, **_k: (0, {})
    )

    user = MagicMock()
    user.id = test_user_id
    user.timezone = "Europe/Amsterdam"
    monkeypatch.setattr(
        "backend.api.external.v1.routes.user_db.get_user_by_id",
        AsyncMock(return_value=user),
    )

    response = client.post(f"/blocks/{block.id}/execute", json={})

    assert response.status_code == 200, f"got {response.status_code}: {response.text}"
    assert "execution_context" in captured
    ctx = captured["execution_context"]
    assert ctx.user_id == test_user_id
    assert ctx.user_timezone == "Europe/Amsterdam"


def test_execute_block_runs_off_the_server_loop_under_the_time_budget(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "backend.executor.direct_block_execution.DIRECT_BLOCK_EXECUTION_TIMEOUT_SECONDS",
        0.5,
    )
    thread_names: list[str] = []
    block = _stub_block(name="SlowBlock")

    async def _never_finishes(_data, **kwargs):
        thread_names.append(threading.current_thread().name)
        await asyncio.Event().wait()
        yield "result", "unreachable"

    block.execute = _never_finishes
    monkeypatch.setattr("backend.blocks.get_block", lambda _: block)
    monkeypatch.setattr(
        "backend.executor.utils.block_usage_cost", lambda *_a, **_k: (0, {})
    )

    response = client.post(f"/blocks/{block.id}/execute", json={})

    assert response.status_code == 504, f"got {response.status_code}: {response.text}"
    assert "did not finish within 0.5 seconds" in response.json()["detail"]
    assert len(thread_names) == 1
    assert thread_names[0].startswith("direct-block-execution")


def test_execute_block_over_the_run_cap_returns_429_before_charging(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "backend.executor.direct_block_execution.DIRECT_BLOCK_EXECUTION_MAX_RUNS_PER_USER",
        0,
    )
    block = _stub_block(name="CappedBlock")
    monkeypatch.setattr("backend.blocks.get_block", lambda _: block)
    charge = AsyncMock()
    monkeypatch.setattr(
        "backend.api.external.v1.routes.charge_for_direct_block_execution", charge
    )

    response = client.post(f"/blocks/{block.id}/execute", json={})

    assert response.status_code == 429, f"got {response.status_code}: {response.text}"
    assert "Wait for one to finish" in response.json()["detail"]
    charge.assert_not_awaited()


def test_execute_block_at_the_retired_worker_cap_returns_503_before_charging(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "backend.executor.direct_block_execution.DIRECT_BLOCK_EXECUTION_MAX_RETIRED_WORKERS",
        0,
    )
    block = _stub_block(name="UnavailableBlock")
    monkeypatch.setattr("backend.blocks.get_block", lambda _: block)
    charge = AsyncMock()
    monkeypatch.setattr(
        "backend.api.external.v1.routes.charge_for_direct_block_execution", charge
    )

    response = client.post(f"/blocks/{block.id}/execute", json={})

    assert response.status_code == 503, f"got {response.status_code}: {response.text}"
    assert "temporarily unavailable" in response.json()["detail"]
    charge.assert_not_awaited()


def test_disabled_block_still_403(monkeypatch: pytest.MonkeyPatch):
    """Pre-existing behavior: disabled blocks return 403, not bypassed by new gates."""
    block = _stub_block(name="DisabledBlock", disabled=True)
    monkeypatch.setattr("backend.blocks.get_block", lambda _: block)

    response = client.post(f"/blocks/{block.id}/execute", json={})

    assert response.status_code == 403
    assert "disabled" in response.json()["detail"]


def test_registry_blocks_needing_a_workflow_are_refused(
    monkeypatch: pytest.MonkeyPatch,
    v1_counter: AsyncMock,
):
    """Check graph-only blocks before charging or running, independent of quota."""
    v1_counter.side_effect = None
    v1_counter.return_value = 1
    ran: list[str] = []

    async def _record_run(self, _data, **_kwargs):
        ran.append(self.name)
        yield "result", "ok"

    charge_mock = AsyncMock()
    monkeypatch.setattr(routes_mod, "charge_for_direct_block_execution", charge_mock)

    outcomes: dict[str, tuple[int, str]] = {}
    for block_id, block_cls in backend.blocks.get_blocks().items():
        block = block_cls()
        if block.disabled or not is_graph_only_block(block):
            continue
        monkeypatch.setattr(block_cls, "execute", _record_run)
        response = client.post(f"/blocks/{block_id}/execute", json={})
        outcomes[block.name] = (response.status_code, response.json().get("detail"))

    assert {
        name: outcome
        for name, outcome in outcomes.items()
        if outcome[0] != 403 or "inside a workflow" not in str(outcome[1])
    } == {}
    assert {"AgentExecutorBlock", "AutoPilotBlock", "OrchestratorBlock"} <= set(
        outcomes
    )
    charge_mock.assert_not_awaited()
    assert not ran


def test_execute_block_context_carries_key_org_and_team(
    monkeypatch: pytest.MonkeyPatch, test_user_id: str
):
    """A team-restricted org key's block run carries that org and team, as
    execute_graph's runs do."""

    async def org_scoped_auth() -> APIAuthorizationInfo:
        return APIAuthorizationInfo(
            user_id=test_user_id,
            scopes=[APIKeyPermission.EXECUTE_BLOCK],
            type="api_key",
            created_at=datetime.now(timezone.utc),
            organization_id="org-a",
            team_id_restriction="team-a",
        )

    app.dependency_overrides[require_auth] = org_scoped_auth
    captured: dict = {}
    block = _stub_block(name="FreeBlock", capture=captured)
    monkeypatch.setattr("backend.blocks.get_block", lambda _: block)
    monkeypatch.setattr(
        "backend.executor.utils.block_usage_cost", lambda *_a, **_k: (0, {})
    )

    response = client.post(f"/blocks/{block.id}/execute", json={})

    assert response.status_code == 200, f"got {response.status_code}: {response.text}"
    ctx = captured["execution_context"]
    assert (ctx.organization_id, ctx.team_id) == ("org-a", "team-a")


def test_unknown_block_returns_404(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("backend.blocks.get_block", lambda _: None)
    response = client.post("/blocks/00000000-0000-0000-0000-deadbeef/execute", json={})
    assert response.status_code == 404


def test_execute_graph_paywall_error_propagates_not_400(
    monkeypatch: pytest.MonkeyPatch,
):
    """The route's broad ``except Exception`` must NOT swallow
    ``UserPaywalledError`` — otherwise the 402 handler registered on
    ``external_api`` never fires and a paywalled user gets a confusing
    400 instead of 402. Locks the explicit re-raise added for this gap.
    """
    monkeypatch.setattr(
        routes_mod,
        "add_graph_execution",
        AsyncMock(side_effect=UserPaywalledError("subscription required")),
    )

    with pytest.raises(UserPaywalledError):
        client.post(
            "/graphs/00000000-0000-0000-0000-000000000001/execute/1",
            json={"node_input": {}},
        )


def test_execute_graph_other_errors_still_return_400(monkeypatch: pytest.MonkeyPatch):
    """Non-paywall exceptions retain the existing 400 behaviour — only
    ``UserPaywalledError`` is the surgical exception to the catch-all."""
    monkeypatch.setattr(
        routes_mod,
        "add_graph_execution",
        AsyncMock(side_effect=ValueError("graph missing")),
    )

    response = client.post(
        "/graphs/00000000-0000-0000-0000-000000000001/execute/1",
        json={"node_input": {}},
    )
    assert response.status_code == 400
    assert "graph missing" in response.json()["detail"]


def test_execute_graph_uses_key_org_not_user_default(
    monkeypatch: pytest.MonkeyPatch, test_user_id: str
):
    """An org-scoped API key must attribute the execution to the KEY's org,
    not the user's default (personal) org. Regression: the route previously
    discarded ``auth.organization_id`` and always resolved via
    ``get_user_default_team`` → multi-org key runs/credits misattributed."""

    async def fake_require_auth() -> APIAuthorizationInfo:
        return APIAuthorizationInfo(
            user_id=test_user_id,
            scopes=list(APIKeyPermission),
            type="api_key",
            created_at=datetime.now(timezone.utc),
            organization_id="org-from-key",
        )

    app.dependency_overrides[require_auth] = fake_require_auth

    monkeypatch.setattr(routes_mod, "enforce_payment_paywall", AsyncMock())
    captured: dict = {}

    async def fake_add(**kwargs):
        captured.update(kwargs)
        return MagicMock(id="exec-1")

    monkeypatch.setattr(routes_mod, "add_graph_execution", fake_add)
    # If the key-org path is honored, the user-default resolver is never hit.
    default_team = AsyncMock(return_value=("personal-org", "personal-team"))
    monkeypatch.setattr(
        "backend.api.features.orgs.db.get_user_default_team", default_team
    )

    response = client.post(
        "/graphs/00000000-0000-0000-0000-000000000001/execute/1",
        json={"node_input": {}},
    )

    assert response.status_code == 200
    assert captured["organization_id"] == "org-from-key"
    assert captured["team_id"] is None
    default_team.assert_not_called()


def test_execute_graph_honors_key_team_restriction(
    monkeypatch: pytest.MonkeyPatch, test_user_id: str
):
    """A team-restricted API key must pin the execution to that team —
    previously ``teamIdRestriction`` was stored but never consumed."""

    async def fake_require_auth() -> APIAuthorizationInfo:
        return APIAuthorizationInfo(
            user_id=test_user_id,
            scopes=list(APIKeyPermission),
            type="api_key",
            created_at=datetime.now(timezone.utc),
            organization_id="org-from-key",
            team_id_restriction="team-from-key",
        )

    app.dependency_overrides[require_auth] = fake_require_auth

    monkeypatch.setattr(routes_mod, "enforce_payment_paywall", AsyncMock())
    captured: dict = {}

    async def fake_add(**kwargs):
        captured.update(kwargs)
        return MagicMock(id="exec-1")

    monkeypatch.setattr(routes_mod, "add_graph_execution", fake_add)

    response = client.post(
        "/graphs/00000000-0000-0000-0000-000000000001/execute/1",
        json={"node_input": {}},
    )

    assert response.status_code == 200
    assert captured["organization_id"] == "org-from-key"
    assert captured["team_id"] == "team-from-key"


def test_execute_graph_block_paywall_propagates(monkeypatch: pytest.MonkeyPatch):
    """``execute_graph_block`` must let ``UserPaywalledError`` propagate
    out so the external app's 402 handler fires. There's no surrounding
    catch-all on this route, so we just lock the propagation."""
    monkeypatch.setattr(
        routes_mod,
        "enforce_payment_paywall",
        AsyncMock(side_effect=UserPaywalledError("subscription required")),
    )

    with pytest.raises(UserPaywalledError):
        client.post(
            "/blocks/00000000-0000-0000-0000-000000000001/execute",
            json={},
        )


@pytest.mark.parametrize(
    "provider, scope",
    [
        ("smtp", None),
        ("smtp", ""),
        ("smtp", "smtp.example.com"),
        ("smtp", "smtp://*.example.com"),
        ("smtp", "postgresql://smtp.example.com"),
        ("database", None),
        ("database", "db.example.com:5432"),
        ("database", "smtp://db.example.com"),
    ],
)
def test_create_login_without_a_valid_scope_is_rejected(
    monkeypatch: pytest.MonkeyPatch, provider: str, scope: str | None
):
    create = AsyncMock()
    monkeypatch.setattr(integrations_mod.creds_manager, "create", create)
    body = {"type": "user_password", "username": "u", "password": "p", "title": "t"}
    if scope is not None:
        body["scope"] = scope

    response = client.post(f"/integrations/{provider}/credentials", json=body)

    assert response.status_code == 400
    create.assert_not_awaited()


@pytest.mark.parametrize(
    "provider, scope",
    [
        ("smtp", "smtp://smtp.example.com:587"),
        ("database", "postgresql://db.example.com:5432"),
        ("database", "mysql://db.example.com"),
    ],
)
def test_create_login_with_a_scope_stores_it(
    monkeypatch: pytest.MonkeyPatch, provider: str, scope: str
):
    create = AsyncMock()
    monkeypatch.setattr(integrations_mod.creds_manager, "create", create)

    response = client.post(
        f"/integrations/{provider}/credentials",
        json={
            "type": "user_password",
            "username": "u",
            "password": "p",
            "title": "t",
            "scope": scope,
        },
    )

    assert response.status_code == 201
    stored = create.await_args.args[1]
    assert stored.provider == provider
    assert stored.scope == scope
