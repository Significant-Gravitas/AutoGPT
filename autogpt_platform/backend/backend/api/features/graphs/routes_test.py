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
from backend.data.graph import GraphModel
from backend.integrations.webhooks.graph_lifecycle_hooks import GraphActivationError

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
