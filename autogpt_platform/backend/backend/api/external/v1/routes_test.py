from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Awaitable
from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import pytest
import pytest_asyncio
from prisma.enums import APIKeyPermission

import backend.api.external.middleware as middleware_mod
import backend.api.external.v1.routes as routes_mod
import backend.api.external.v1.tools as tools_mod
from backend.api.external.middleware import require_auth
from backend.api.external.v1.routes import v1_router
from backend.copilot.rate_limit import UserPaywalledError
from backend.copilot.response_model import StreamToolOutputAvailable
from backend.copilot.tools.models import ExecutionStartedResponse
from backend.data.auth.base import APIAuthorizationInfo
from backend.util.exceptions import InsufficientBalanceError

app = fastapi.FastAPI()
app.include_router(v1_router)
client = fastapi.testclient.TestClient(app)


class _PassthroughLeaseGuard:
    async def run(self, action: Awaitable[Any]) -> Any:
        return await action


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def server():
    return None


@pytest_asyncio.fixture(scope="session", loop_scope="session", autouse=True)
async def graph_cleanup():
    yield


@pytest.fixture(autouse=True)
def setup_auth(test_user_id):
    """Override require_auth to return a synthetic API-key principal with all scopes."""

    async def fake_require_auth() -> APIAuthorizationInfo:
        return APIAuthorizationInfo(
            user_id=test_user_id,
            scopes=list(APIKeyPermission),
            type="api_key",
            created_at=datetime.now(timezone.utc),
            organization_id="test-org",
            team_id_restriction="test-team",
        )

    app.dependency_overrides[require_auth] = fake_require_auth
    yield
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def live_resource_access(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "backend.api.external.middleware.has_live_resource_access",
        AsyncMock(return_value=True),
    )


@pytest.fixture(autouse=True)
def resource_barriers(monkeypatch: pytest.MonkeyPatch):
    @asynccontextmanager
    async def allow(*_args, **_kwargs):
        yield True

    monkeypatch.setattr(routes_mod, "live_resource_access_barrier", allow)
    monkeypatch.setattr(routes_mod, "agent_graph_attachment_barrier", allow)
    monkeypatch.setattr(middleware_mod, "_live_authorization_principal", allow)


@pytest.fixture(autouse=True)
def exact_external_graph_scope(monkeypatch: pytest.MonkeyPatch):
    async def exact_library_agent(**kwargs):
        return MagicMock(team_id=kwargs["team_id_restriction"])

    monkeypatch.setattr(
        "backend.api.features.library.db.get_library_agent_by_graph_id",
        exact_library_agent,
    )


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
    block.execution_timeout_seconds = 30

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


def test_disabled_block_still_403(monkeypatch: pytest.MonkeyPatch):
    """Pre-existing behavior: disabled blocks return 403, not bypassed by new gates."""
    block = _stub_block(name="DisabledBlock", disabled=True)
    monkeypatch.setattr("backend.blocks.get_block", lambda _: block)

    response = client.post(f"/blocks/{block.id}/execute", json={})

    assert response.status_code == 403
    assert "disabled" in response.json()["detail"]


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


def test_execute_graph_org_home_key_rejects_team_only_graph(
    monkeypatch: pytest.MonkeyPatch, test_user_id: str
):
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
    exact_lookup = AsyncMock(return_value=None)
    resolve_grant = AsyncMock()
    add_execution = AsyncMock()
    monkeypatch.setattr(
        "backend.api.features.library.db.get_library_agent_by_graph_id",
        exact_lookup,
    )
    monkeypatch.setattr(routes_mod, "resolve_graph_grant", resolve_grant)
    monkeypatch.setattr(routes_mod, "add_graph_execution", add_execution)

    response = client.post(
        "/graphs/00000000-0000-0000-0000-000000000001/execute/1",
        json={"node_input": {}},
    )

    assert response.status_code == 404
    exact_lookup.assert_awaited_once_with(
        user_id=test_user_id,
        graph_id="00000000-0000-0000-0000-000000000001",
        graph_version=1,
        organization_id="org-from-key",
        team_id_restriction=None,
        exact_scope=True,
    )
    resolve_grant.assert_not_awaited()
    add_execution.assert_not_awaited()


def test_execute_graph_team_key_accepts_exact_team_grant(
    monkeypatch: pytest.MonkeyPatch, test_user_id: str
):
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
    monkeypatch.setattr(
        "backend.api.features.library.db.get_library_agent_by_graph_id",
        AsyncMock(return_value=None),
    )
    grant = MagicMock(principalId="team-from-key")
    resolve_grant = AsyncMock(return_value=grant)
    add_execution = AsyncMock(return_value=MagicMock(id="exec-1"))
    monkeypatch.setattr(routes_mod, "resolve_graph_grant", resolve_grant)
    monkeypatch.setattr(routes_mod, "add_graph_execution", add_execution)

    response = client.post(
        "/graphs/00000000-0000-0000-0000-000000000001/execute/1",
        json={"node_input": {}},
    )

    assert response.status_code == 200
    resolve_grant.assert_awaited_once_with(
        test_user_id,
        "00000000-0000-0000-0000-000000000001",
        capability=routes_mod.GrantCapability.EXECUTE,
        graph_version=1,
        organization_id="org-from-key",
        team_id_restriction="team-from-key",
    )
    assert add_execution.await_args.kwargs["organization_id"] == "org-from-key"
    assert add_execution.await_args.kwargs["team_id"] == "team-from-key"


def test_run_agent_tool_honors_key_team_restriction(
    monkeypatch: pytest.MonkeyPatch, test_user_id: str
):
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
    captured: dict = {}

    async def fake_execute(**kwargs):
        captured.update(kwargs)
        return StreamToolOutputAvailable(
            toolCallId=kwargs["tool_call_id"],
            toolName="run_agent",
            output={"ok": True},
        )

    monkeypatch.setattr(tools_mod.run_agent_tool, "execute", fake_execute)

    response = client.post(
        "/tools/run-agent",
        json={"username_agent_slug": "test/agent"},
    )

    assert response.status_code == 200
    assert captured["session"].organization_id == "org-from-key"
    assert captured["session"].team_id == "team-from-key"


def test_external_scheduled_run_holds_execute_and_create_leases(
    monkeypatch: pytest.MonkeyPatch, test_user_id: str
):
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
    active: list[str] = []

    @asynccontextmanager
    async def lease(user_id, organization_id, team_id, access):
        assert (user_id, organization_id, team_id) == (
            test_user_id,
            "org-from-key",
            "team-from-key",
        )
        active.append(access)
        try:
            yield _PassthroughLeaseGuard()
        finally:
            active.remove(access)

    async def fake_execute(user_id, session, **_kwargs):
        assert user_id == test_user_id
        assert sorted(active) == ["create", "execute"]
        return ExecutionStartedResponse(
            message="Scheduled",
            session_id=session.session_id,
            execution_id="schedule-1",
            graph_id="graph-1",
            graph_name="Agent",
            status="SCHEDULED",
        )

    monkeypatch.setattr("backend.copilot.tools.base.live_resource_lease", lease)
    monkeypatch.setattr(tools_mod.run_agent_tool, "_execute", fake_execute)

    response = client.post(
        "/tools/run-agent",
        json={
            "username_agent_slug": "test/agent",
            "schedule_name": "Daily",
            "cron": "0 9 * * *",
        },
    )

    assert response.status_code == 200
    assert response.json()["execution_id"] == "schedule-1"
    assert active == []


@pytest.mark.asyncio
async def test_execution_results_scope_lookup_to_key_org_and_team(
    monkeypatch: pytest.MonkeyPatch, test_user_id: str
):
    lookup = AsyncMock(return_value=None)
    monkeypatch.setattr(routes_mod.execution_db, "get_graph_execution", lookup)
    auth = APIAuthorizationInfo(
        user_id=test_user_id,
        scopes=list(APIKeyPermission),
        type="api_key",
        created_at=datetime.now(timezone.utc),
        organization_id="org-from-key",
        team_id_restriction="team-from-key",
    )

    with pytest.raises(fastapi.HTTPException) as exc:
        await routes_mod.get_graph_execution_results(
            "graph-other-org",
            "execution-other-org",
            auth,
        )

    assert exc.value.status_code == 404
    lookup.assert_awaited_once_with(
        user_id=test_user_id,
        execution_id="execution-other-org",
        include_node_executions=True,
        organization_id="org-from-key",
        team_id_restriction="team-from-key",
    )


@pytest.mark.asyncio
async def test_execution_results_reject_mismatched_path_graph(
    monkeypatch: pytest.MonkeyPatch, test_user_id: str
):
    execution = MagicMock(graph_id="graph-real", graph_version=1)
    monkeypatch.setattr(
        routes_mod.execution_db,
        "get_graph_execution",
        AsyncMock(return_value=execution),
    )
    graph_lookup = AsyncMock()
    monkeypatch.setattr(routes_mod.graph_db, "get_graph", graph_lookup)
    auth = APIAuthorizationInfo(
        user_id=test_user_id,
        scopes=list(APIKeyPermission),
        type="api_key",
        created_at=datetime.now(timezone.utc),
        organization_id="org-from-key",
    )

    with pytest.raises(fastapi.HTTPException) as exc:
        await routes_mod.get_graph_execution_results(
            "graph-wrong",
            "execution-1",
            auth,
        )

    assert exc.value.status_code == 404
    graph_lookup.assert_not_called()


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
