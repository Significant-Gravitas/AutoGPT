"""Tests for the graph routes."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, Mock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from fastapi.routing import APIRoute
from pytest_snapshot.plugin import Snapshot

from backend.api.features.graphs.routes import router
from backend.api.rest_api import app as real_app
from backend.api.rest_api import handle_internal_http_error
from backend.data import execution as execution_db
from backend.data.execution import ExecutionStatus
from backend.data.graph import GraphModel
from backend.data.onboarding import OnboardingStep
from backend.integrations.webhooks.graph_lifecycle_hooks import GraphActivationError
from backend.util.exceptions import GraphValidationError

app = fastapi.FastAPI()
app.include_router(router)
# Mirror rest_api.py's GraphActivationError -> 400 mapping so the atomicity
# tests verify the same behaviour the real app exposes.
app.add_exception_handler(GraphActivationError, handle_internal_http_error(400))
client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user, test_user_id):
    from autogpt_libs.auth.dependencies import get_request_context
    from autogpt_libs.auth.jwt_utils import get_jwt_payload
    from autogpt_libs.auth.models import RequestContext

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]

    async def _fake_request_context() -> RequestContext:
        return RequestContext(
            user_id=test_user_id,
            org_id="test-org",
            team_id=None,
            is_org_owner=True,
            is_org_admin=True,
            is_org_billing_manager=False,
            is_team_admin=True,
            is_team_billing_manager=False,
            seat_status="ACTIVE",
        )

    app.dependency_overrides[get_request_context] = _fake_request_context
    yield
    app.dependency_overrides.clear()


def _execution_meta() -> execution_db.GraphExecutionMeta:
    return execution_db.GraphExecutionMeta(
        id="exec-1",
        user_id="user-1",
        graph_id="graph-1",
        graph_version=1,
        inputs={},
        credential_inputs=None,
        nodes_input_masks=None,
        preset_id=None,
        status=ExecutionStatus.QUEUED,
        stats=None,
    )


EXPECTED_OPERATIONS = {
    ("get", "/api/graphs"): "getV1List user graphs",
    ("post", "/api/graphs"): "postV1Create new graph",
    ("get", "/api/graphs/{graph_id}"): "getV1Get specific graph",
    ("put", "/api/graphs/{graph_id}"): "putV1Update graph version",
    ("delete", "/api/graphs/{graph_id}"): "deleteV1Delete graph permanently",
    ("get", "/api/graphs/{graph_id}/versions"): "getV1Get all graph versions",
    ("get", "/api/graphs/{graph_id}/versions/{version}"): "getV1Get graph version",
    ("put", "/api/graphs/{graph_id}/versions/active"): "putV1Set active graph version",
    ("patch", "/api/graphs/{graph_id}/settings"): "patchV1Update graph settings",
    ("post", "/api/graphs/{graph_id}/execute/{graph_version}"): (
        "postV1Execute graph agent"
    ),
}


@pytest.mark.parametrize(
    "method,path,operation_id",
    [(m, p, oid) for (m, p), oid in EXPECTED_OPERATIONS.items()],
)
def test_graph_operation_is_published(method: str, path: str, operation_id: str):
    operation = real_app.openapi()["paths"][path][method]
    assert operation["operationId"] == operation_id
    assert operation["tags"] == ["v1", "graphs"]
    # All ten are authenticated, and the dependency now lives on the router
    # rather than on each route — so nothing else would notice it going missing.
    assert "security" in operation


@pytest.mark.parametrize("path", sorted({p for _, p in EXPECTED_OPERATIONS}))
def test_graph_route_requires_an_authenticated_user(path: str):
    """`security` in the spec does not prove this: each handler's own
    Security(get_user_id) puts it there, so removing the router's
    requires_user leaves the schema unchanged. Assert the dependency."""
    for route in real_app.routes:
        if isinstance(route, APIRoute) and route.path == path:
            assert "requires_user" in {
                d.call.__name__ for d in route.dependant.dependencies if d.call
            }


def test_execute_graph_is_behind_the_payment_paywall():
    """The only route with a dependency beyond auth, and the one whose loss
    would be silent: it gates spending, and no other test exercises it."""
    route = next(
        r
        for r in real_app.routes
        if isinstance(r, APIRoute)
        and r.path == "/api/graphs/{graph_id}/execute/{graph_version}"
    )
    assert "enforce_payment_paywall" in {
        d.call.__name__ for d in route.dependant.dependencies if d.call
    }


def test_graph_surface_has_no_other_operations():
    served = {
        (method.lower(), route.path)
        for route in real_app.routes
        if isinstance(route, APIRoute)
        and route.endpoint.__module__ == "backend.api.features.graphs.routes"
        for method in route.methods
        if method != "HEAD"
    }
    assert served == set(EXPECTED_OPERATIONS)


# /api/graphs/{graph_id}/* is now served by three modules. Every fourth segment
# is literal, so order between them is unobservable - this is what keeps it so.
@pytest.mark.parametrize(
    "path,module",
    [
        ("/api/graphs/{graph_id}", "backend.api.features.graphs.routes"),
        (
            "/api/graphs/{graph_id}/execute/{graph_version}",
            "backend.api.features.graphs.routes",
        ),
        ("/api/graphs/{graph_id}/settings", "backend.api.features.graphs.routes"),
        ("/api/graphs/{graph_id}/versions", "backend.api.features.graphs.routes"),
        (
            "/api/graphs/{graph_id}/versions/active",
            "backend.api.features.graphs.routes",
        ),
        (
            "/api/graphs/{graph_id}/versions/{version}",
            "backend.api.features.graphs.routes",
        ),
        ("/api/graphs/{graph_id}/schedules", "backend.api.features.schedules.routes"),
        (
            "/api/graphs/{graph_id}/executions",
            "backend.api.features.executions.routes",
        ),
        (
            "/api/graphs/{graph_id}/executions/{graph_exec_id}",
            "backend.api.features.executions.routes",
        ),
        (
            "/api/graphs/{graph_id}/executions/{graph_exec_id}/stop",
            "backend.api.features.executions.routes",
        ),
        (
            "/api/graphs/{graph_id}/executions/{graph_exec_id}/share",
            "backend.api.features.executions.routes",
        ),
    ],
)
def test_graph_subpath_is_served_by_the_expected_module(path: str, module: str):
    handlers = {
        route.endpoint.__module__
        for route in real_app.routes
        if isinstance(route, APIRoute) and route.path == path
    }
    assert handlers == {module}


def test_agent_server_helpers_reference_this_module():
    """rest_api.py's AgentServer test helpers call these five directly, and the
    whole backend suite reaches graphs through them."""
    import backend.api.rest_api as rest_api

    for name in (
        "execute_graph",
        "get_graph",
        "CreateGraph",
        "create_new_graph",
        "delete_graph",
    ):
        assert getattr(rest_api.graphs_routes, name) is not None


def test_get_graphs(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
    test_user_id: str,
) -> None:
    """Test get graphs endpoint"""
    mock_graph = GraphModel(
        id="graph-123",
        version=1,
        is_active=True,
        name="Test Graph",
        description="A test graph",
        user_id=test_user_id,
        created_at=datetime(2025, 9, 4, 13, 37),
    )

    mocker.patch(
        "backend.data.graph.list_graphs_paginated",
        return_value=Mock(graphs=[mock_graph]),
    )

    response = client.get("/graphs")

    assert response.status_code == 200
    response_data = response.json()
    assert len(response_data) == 1
    assert response_data[0]["id"] == "graph-123"

    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "grphs_all",
    )


def test_get_graph(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
    test_user_id: str,
) -> None:
    """Test get single graph endpoint"""
    mock_graph = GraphModel(
        id="graph-123",
        version=1,
        is_active=True,
        name="Test Graph",
        description="A test graph",
        user_id=test_user_id,
        created_at=datetime(2025, 9, 4, 13, 37),
    )

    mocker.patch(
        "backend.api.features.graphs.routes.graph_db.get_graph",
        return_value=mock_graph,
    )

    response = client.get("/graphs/graph-123")

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["id"] == "graph-123"

    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "grph_single",
    )


def test_get_graph_not_found(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Test get graph with non-existent ID"""
    mocker.patch(
        "backend.api.features.graphs.routes.graph_db.get_graph",
        return_value=None,
    )

    response = client.get("/graphs/nonexistent-graph")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_delete_graph(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
    test_user_id: str,
) -> None:
    """Test delete graph endpoint"""
    # Mock active graph for deactivation
    mock_graph = GraphModel(
        id="graph-123",
        version=1,
        is_active=True,
        name="Test Graph",
        description="A test graph",
        user_id=test_user_id,
        created_at=datetime(2025, 9, 4, 13, 37),
    )

    mocker.patch(
        "backend.api.features.graphs.routes.graph_db.get_graph",
        return_value=mock_graph,
    )
    mocker.patch(
        "backend.api.features.graphs.routes.on_graph_deactivate",
        return_value=None,
    )
    mocker.patch(
        "backend.api.features.graphs.routes.graph_db.delete_graph",
        return_value=3,  # Number of versions deleted
    )

    response = client.delete("/graphs/graph-123")

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["version_counts"] == 3

    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "grphs_del",
    )


def test_create_new_graph_returns_400_and_persists_nothing_on_activation_error(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Core atomicity guarantee: when before_graph_activate raises,
    POST /graphs must return 400 and never call create_graph / create_library_agent.
    Reordering activation back to post-save would break this test."""
    from backend.integrations.webhooks.graph_lifecycle_hooks import GraphActivationError

    mock_graph_model = Mock()
    mocker.patch(
        "backend.api.features.graphs.routes.graph_db.make_graph_model",
        return_value=mock_graph_model,
    )
    activate_mock = mocker.patch(
        "backend.api.features.graphs.routes.before_graph_activate",
        new=AsyncMock(
            side_effect=GraphActivationError(
                "Credential #cred-1 needs reconnect — please reconnect"
            )
        ),
    )
    create_graph_mock = mocker.patch(
        "backend.api.features.graphs.routes.graph_db.create_graph", new=AsyncMock()
    )
    create_lib_agent_mock = mocker.patch(
        "backend.api.features.graphs.routes.library_db.create_library_agent",
        new=AsyncMock(),
    )

    response = client.post(
        "/graphs",
        json={
            "graph": {
                "name": "Test Graph",
                "description": "Test",
                "nodes": [],
                "links": [],
            }
        },
    )

    assert response.status_code == 400
    assert "reconnect" in response.json()["detail"]
    activate_mock.assert_awaited_once()
    create_graph_mock.assert_not_awaited()
    create_lib_agent_mock.assert_not_awaited()


def test_update_graph_returns_400_and_persists_nothing_on_activation_error(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Same atomicity guarantee on PUT /graphs/{id}: an activation failure
    must short-circuit with 400 before any new graph version is written."""
    from backend.integrations.webhooks.graph_lifecycle_hooks import GraphActivationError

    mock_graph_model = Mock(is_active=True)
    existing_version = Mock(version=1, is_active=True)
    mocker.patch(
        "backend.api.features.graphs.routes.graph_db.get_graph_all_versions",
        new=AsyncMock(return_value=[existing_version]),
    )
    mocker.patch(
        "backend.api.features.graphs.routes.graph_db.make_graph_model",
        return_value=mock_graph_model,
    )
    activate_mock = mocker.patch(
        "backend.api.features.graphs.routes.before_graph_activate",
        new=AsyncMock(
            side_effect=GraphActivationError(
                "Credential #cred-1 needs reconnect — please reconnect"
            )
        ),
    )
    create_graph_mock = mocker.patch(
        "backend.api.features.graphs.routes.graph_db.create_graph", new=AsyncMock()
    )
    update_lib_agent_mock = mocker.patch(
        "backend.api.features.graphs.routes.library_db.update_library_agent_version_and_settings",
        new=AsyncMock(),
    )

    response = client.put(
        "/graphs/graph-123",
        json={
            "id": "graph-123",
            "name": "Test Graph",
            "description": "Test",
            "nodes": [],
            "links": [],
        },
    )

    assert response.status_code == 400
    assert "reconnect" in response.json()["detail"]
    activate_mock.assert_awaited_once()
    create_graph_mock.assert_not_awaited()
    update_lib_agent_mock.assert_not_awaited()


def test_get_graph_all_versions_returns_404_when_empty(
    mocker: pytest_mock.MockFixture,
) -> None:
    mocker.patch(
        "backend.api.features.graphs.routes.graph_db.get_graph_all_versions",
        AsyncMock(return_value=[]),
    )

    response = client.get("/graphs/graph-1/versions")

    assert response.status_code == 404


def test_update_graph_settings_requires_a_library_agent(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Settings live on the caller's library agent, so a graph the caller has
    not added must not be settable."""
    mocker.patch(
        "backend.api.features.graphs.routes.library_db.get_library_agent_by_graph_id",
        AsyncMock(return_value=None),
    )

    response = client.patch("/graphs/graph-1/settings", json={})

    assert response.status_code == 404


def test_update_graph_settings_scopes_the_write_to_the_caller(
    mocker: pytest_mock.MockFixture,
    test_user_id: str,
) -> None:
    mocker.patch(
        "backend.api.features.graphs.routes.library_db.get_library_agent_by_graph_id",
        AsyncMock(return_value=Mock(id="lib-1")),
    )
    update = mocker.patch(
        "backend.api.features.graphs.routes.library_db.update_library_agent",
        AsyncMock(return_value=Mock(settings={})),
    )

    response = client.patch("/graphs/graph-1/settings", json={})

    assert response.status_code == 200
    kwargs = update.await_args.kwargs
    assert kwargs["library_agent_id"] == "lib-1"
    assert kwargs["user_id"] == test_user_id


def test_execute_graph_refuses_a_zero_balance(
    mocker: pytest_mock.MockFixture,
) -> None:
    """The balance gate is the last thing between an empty account and a paid
    run; a 402 here is what stops it."""
    credit_model = Mock()
    credit_model.get_credits = AsyncMock(return_value=0)
    mocker.patch(
        "backend.api.features.graphs.routes.get_credit_model",
        AsyncMock(return_value=credit_model),
    )
    started = mocker.patch(
        "backend.api.features.graphs.routes.execution_utils.add_graph_execution",
        AsyncMock(),
    )

    response = client.post("/graphs/graph-1/execute/1", json={})

    assert response.status_code == 402
    started.assert_not_awaited()


def test_execute_graph_skips_the_balance_check_for_a_dry_run(
    mocker: pytest_mock.MockFixture,
) -> None:
    """A dry run costs nothing, so it must not be blocked by an empty balance
    — and must not consult the credit model at all."""
    credit = mocker.patch(
        "backend.api.features.graphs.routes.get_credit_model", AsyncMock()
    )
    mocker.patch(
        "backend.api.features.graphs.routes.execution_utils.add_graph_execution",
        AsyncMock(return_value=_execution_meta()),
    )
    mocker.patch("backend.api.features.graphs.routes.record_graph_operation", Mock())

    response = client.post("/graphs/graph-1/execute/1", json={"dry_run": True})

    assert response.status_code == 200
    credit.assert_not_awaited()


def test_execute_graph_forwards_the_org_and_team_scope(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Dropping either would run the graph outside the caller's tenancy."""
    credit_model = Mock()
    credit_model.get_credits = AsyncMock(return_value=100)
    mocker.patch(
        "backend.api.features.graphs.routes.get_credit_model",
        AsyncMock(return_value=credit_model),
    )
    started = mocker.patch(
        "backend.api.features.graphs.routes.execution_utils.add_graph_execution",
        AsyncMock(return_value=_execution_meta()),
    )
    mocker.patch("backend.api.features.graphs.routes.record_graph_operation", Mock())

    response = client.post("/graphs/graph-1/execute/2", json={})

    assert response.status_code == 200
    kwargs = started.await_args.kwargs
    assert kwargs["organization_id"] == "test-org"
    assert kwargs["graph_version"] == 2


def test_execute_graph_marks_the_onboarding_step_for_a_library_run(
    mocker: pytest_mock.MockFixture,
) -> None:
    """The step is keyed off `source`, so a run started from the library is the
    only thing that can complete it."""
    credit_model = Mock()
    credit_model.get_credits = AsyncMock(return_value=100)
    mocker.patch(
        "backend.api.features.graphs.routes.get_credit_model",
        AsyncMock(return_value=credit_model),
    )
    mocker.patch(
        "backend.api.features.graphs.routes.execution_utils.add_graph_execution",
        AsyncMock(return_value=_execution_meta()),
    )
    mocker.patch("backend.api.features.graphs.routes.record_graph_operation", Mock())
    step = mocker.patch(
        "backend.api.features.graphs.routes.complete_onboarding_step", AsyncMock()
    )

    response = client.post("/graphs/graph-1/execute/1", json={"source": "library"})

    assert response.status_code == 200
    assert step.await_args.args[1] is OnboardingStep.LIBRARY_RUN_AGENT


def test_execute_graph_returns_structured_validation_errors(
    mocker: pytest_mock.MockFixture,
) -> None:
    """The builder parses `node_errors` to highlight the offending nodes; a
    plain 500 would lose that."""
    credit_model = Mock()
    credit_model.get_credits = AsyncMock(return_value=100)
    mocker.patch(
        "backend.api.features.graphs.routes.get_credit_model",
        AsyncMock(return_value=credit_model),
    )
    mocker.patch(
        "backend.api.features.graphs.routes.execution_utils.add_graph_execution",
        AsyncMock(
            side_effect=GraphValidationError(
                message="bad graph", node_errors={"node-1": "missing input"}
            )
        ),
    )
    mocker.patch("backend.api.features.graphs.routes.record_graph_operation", Mock())

    response = client.post("/graphs/graph-1/execute/1", json={})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["type"] == "validation_error"
    assert detail["node_errors"] == {"node-1": "missing input"}


def test_update_graph_rejects_an_id_that_contradicts_the_uri(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Without this the body's id would silently win over the URI's."""
    response = client.put(
        "/graphs/graph-1",
        json={
            "id": "graph-2",
            "name": "x",
            "description": "",
            "nodes": [],
            "links": [],
        },
    )

    assert response.status_code == 400


def test_update_graph_returns_404_for_an_unknown_graph(
    mocker: pytest_mock.MockFixture,
) -> None:
    mocker.patch(
        "backend.api.features.graphs.routes.graph_db.get_graph_all_versions",
        AsyncMock(return_value=[]),
    )

    response = client.put(
        "/graphs/graph-1",
        json={
            "id": "graph-1",
            "name": "x",
            "description": "",
            "nodes": [],
            "links": [],
        },
    )

    assert response.status_code == 404


def test_create_new_graph_reassigns_ids_and_persists_in_order(
    mocker: pytest_mock.MockFixture,
) -> None:
    """The graph gets fresh ids before anything is written, and the library
    agent is created only after the graph — the same ordering the activation
    -error test above pins from the failure side. Reassignment matters because
    a submitted id would otherwise let a caller collide with someone else's."""
    calls: list[str] = []
    mocker.patch(
        "backend.api.features.graphs.routes.before_graph_activate",
        new=AsyncMock(side_effect=lambda g, **_: g),
    )
    mocker.patch(
        "backend.api.features.graphs.routes.graph_db.create_graph",
        new=AsyncMock(side_effect=lambda *a, **k: calls.append("graph")),
    )
    mocker.patch(
        "backend.api.features.graphs.routes.library_db.create_library_agent",
        new=AsyncMock(side_effect=lambda *a, **k: calls.append("library")),
    )

    response = client.post(
        "/graphs",
        json={
            "graph": {
                "id": "submitted-id",
                "name": "x",
                "description": "",
                "nodes": [],
                "links": [],
            }
        },
    )

    assert response.status_code == 200
    assert calls == ["graph", "library"]
    assert response.json()["id"] != "submitted-id"


def test_set_active_version_returns_404_for_an_unknown_version(
    mocker: pytest_mock.MockFixture,
) -> None:
    """The lookup is scoped to the caller, so a version belonging to someone
    else reads as absent rather than being activated."""
    lookup = mocker.patch(
        "backend.api.features.graphs.routes.graph_db.get_graph",
        AsyncMock(return_value=None),
    )
    activate = mocker.patch(
        "backend.api.features.graphs.routes.graph_db.set_graph_active_version",
        AsyncMock(),
    )

    response = client.put(
        "/graphs/graph-1/versions/active", json={"active_graph_version": 7}
    )

    assert response.status_code == 404
    activate.assert_not_awaited()
    assert lookup.await_args.kwargs["user_id"]


def test_get_graph_all_versions_scopes_to_the_caller_org(
    mocker: pytest_mock.MockFixture,
) -> None:
    lookup = mocker.patch(
        "backend.api.features.graphs.routes.graph_db.get_graph_all_versions",
        AsyncMock(return_value=[]),
    )

    client.get("/graphs/graph-1/versions")

    assert lookup.await_args.kwargs["organization_id"] == "test-org"


def test_execute_graph_marks_the_builder_onboarding_step(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Sibling of the library case: the two sources complete different steps,
    so collapsing them would silently mark the wrong one."""
    credit_model = Mock()
    credit_model.get_credits = AsyncMock(return_value=100)
    mocker.patch(
        "backend.api.features.graphs.routes.get_credit_model",
        AsyncMock(return_value=credit_model),
    )
    mocker.patch(
        "backend.api.features.graphs.routes.execution_utils.add_graph_execution",
        AsyncMock(return_value=_execution_meta()),
    )
    mocker.patch("backend.api.features.graphs.routes.record_graph_operation", Mock())
    step = mocker.patch(
        "backend.api.features.graphs.routes.complete_onboarding_step", AsyncMock()
    )

    response = client.post("/graphs/graph-1/execute/1", json={"source": "builder"})

    assert response.status_code == 200
    assert step.await_args.args[1] is OnboardingStep.BUILDER_RUN_AGENT


def test_execute_graph_records_a_failure_and_re_raises(
    mocker: pytest_mock.MockFixture,
) -> None:
    """An unexpected error must still be counted; swallowing it would make the
    execute metric read as if nothing went wrong."""
    credit_model = Mock()
    credit_model.get_credits = AsyncMock(return_value=100)
    mocker.patch(
        "backend.api.features.graphs.routes.get_credit_model",
        AsyncMock(return_value=credit_model),
    )
    mocker.patch(
        "backend.api.features.graphs.routes.execution_utils.add_graph_execution",
        AsyncMock(side_effect=RuntimeError("boom")),
    )
    record = mocker.patch(
        "backend.api.features.graphs.routes.record_graph_operation", Mock()
    )

    with pytest.raises(RuntimeError):
        client.post("/graphs/graph-1/execute/1", json={})

    assert record.call_args.kwargs["status"] == "error"
