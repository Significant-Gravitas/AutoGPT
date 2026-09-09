"""Tests for the execution routes."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from fastapi.routing import APIRoute

from backend.api.features.executions.routes import router
from backend.api.rest_api import app as real_app

app = fastapi.FastAPI()
app.include_router(router)
client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


# The ten operations this module owns, with the tag list and auth posture each
# publishes. Two are deliberately unauthenticated and three carry no tag of
# their own, so neither can be hoisted to the router.
EXPECTED_OPERATIONS = {
    ("get", "/api/executions"): (["v1", "graphs"], True),
    ("get", "/api/executions/cost-summary"): (["v1", "graphs"], True),
    ("delete", "/api/executions/{graph_exec_id}"): (["v1", "graphs"], True),
    ("get", "/api/graphs/{graph_id}/executions"): (["v1", "graphs"], True),
    ("get", "/api/graphs/{graph_id}/executions/{graph_exec_id}"): (
        ["v1", "graphs"],
        True,
    ),
    ("post", "/api/graphs/{graph_id}/executions/{graph_exec_id}/stop"): (
        ["v1", "graphs"],
        True,
    ),
    ("post", "/api/graphs/{graph_id}/executions/{graph_exec_id}/share"): (["v1"], True),
    ("delete", "/api/graphs/{graph_id}/executions/{graph_exec_id}/share"): (
        ["v1"],
        True,
    ),
    ("get", "/api/public/shared/{share_token}"): (["v1"], False),
    ("get", "/api/public/shared/{share_token}/files/{file_id}/download"): (
        ["v1", "graphs"],
        False,
    ),
}


@pytest.mark.parametrize(
    "method,path,tags,authenticated",
    [(m, p, t, a) for (m, p), (t, a) in EXPECTED_OPERATIONS.items()],
)
def test_execution_operation_is_published(
    method: str, path: str, tags: list[str], authenticated: bool
):
    operation = real_app.openapi()["paths"][path][method]
    assert operation["tags"] == tags
    assert ("security" in operation) is authenticated


def test_execution_surface_has_no_other_operations():
    """Keyed on the owning module, not a path prefix: /api/executions/admin/* and
    /api/public/shared/chats/* belong to other modules and share these prefixes."""
    served = {
        (method.lower(), route.path)
        for route in real_app.routes
        if isinstance(route, APIRoute)
        and route.endpoint.__module__ == "backend.api.features.executions.routes"
        for method in route.methods
        if method != "HEAD"
    }
    assert served == set(EXPECTED_OPERATIONS)


@pytest.mark.parametrize("path", sorted({p for _, p in EXPECTED_OPERATIONS}))
def test_execution_path_is_served_by_this_module(path: str):
    handlers = {
        route.endpoint.__module__
        for route in real_app.routes
        if isinstance(route, APIRoute) and route.path == path
    }
    assert handlers == {"backend.api.features.executions.routes"}


# The three /graphs/{graph_id}/executions* routes now register from this module
# while the other /graphs/{graph_id}/X routes stay in v1; a wildcard fourth
# segment on either side would silently swallow the other.
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
        (
            "/api/graphs/{graph_id}/schedules",
            "backend.api.features.schedules.routes",
        ),
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


def test_executions_cost_summary_returns_payload(
    mocker: pytest_mock.MockFixture,
) -> None:
    """The /executions/cost-summary route returns the aggregated payload."""
    from prisma.enums import AgentExecutionStatus

    from backend.data.execution_cost_summary import (
        UserAgentCostRollup,
        UserDailyCost,
        UserExecutionCostSummary,
        UserTopRun,
    )

    summary = UserExecutionCostSummary(
        total_cents=4200,
        run_count=12,
        billable_run_count=10,
        failed_cost_cents=500,
        by_agent=[
            UserAgentCostRollup(graph_id="g-1", cost_cents=3000, run_count=8),
            UserAgentCostRollup(graph_id="g-2", cost_cents=1200, run_count=4),
        ],
        top_runs=[
            UserTopRun(
                execution_id="exec-1",
                graph_id="g-1",
                cost_cents=2500,
                started_at=datetime(2026, 5, 10, 12, 0, tzinfo=timezone.utc),
                status=AgentExecutionStatus.COMPLETED,
                duration_seconds=45.5,
                node_error_count=0,
            ),
        ],
        daily=[
            UserDailyCost(date="2026-05-10", cost_cents=3000, run_count=8),
            UserDailyCost(date="2026-05-11", cost_cents=1200, run_count=4),
        ],
    )

    mock_fn = mocker.patch(
        "backend.api.features.executions.routes.get_user_cost_summary",
        AsyncMock(return_value=summary),
    )

    response = client.get("/executions/cost-summary")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_cents"] == 4200
    assert payload["run_count"] == 12
    assert payload["billable_run_count"] == 10
    assert payload["failed_cost_cents"] == 500
    assert len(payload["by_agent"]) == 2
    assert payload["by_agent"][0]["graph_id"] == "g-1"
    assert payload["by_agent"][0]["cost_cents"] == 3000
    assert len(payload["top_runs"]) == 1
    assert payload["top_runs"][0]["execution_id"] == "exec-1"
    assert payload["top_runs"][0]["cost_cents"] == 2500
    assert len(payload["daily"]) == 2
    assert payload["daily"][0]["date"] == "2026-05-10"
    mock_fn.assert_awaited_once()


def test_executions_cost_summary_forwards_since_until(
    mocker: pytest_mock.MockFixture,
    test_user_id: str,
) -> None:
    """since/until query params should reach get_user_cost_summary."""
    from backend.data.execution_cost_summary import UserExecutionCostSummary

    mock_fn = mocker.patch(
        "backend.api.features.executions.routes.get_user_cost_summary",
        AsyncMock(
            return_value=UserExecutionCostSummary(
                total_cents=0,
                run_count=0,
                billable_run_count=0,
                failed_cost_cents=0,
                by_agent=[],
                top_runs=[],
                daily=[],
            )
        ),
    )

    response = client.get(
        "/executions/cost-summary"
        "?since=2026-05-01T00:00:00Z"
        "&until=2026-05-15T00:00:00Z"
        "&top_runs_limit=5"
    )

    assert response.status_code == 200
    kwargs = mock_fn.await_args.kwargs
    assert kwargs["user_id"] == test_user_id
    assert kwargs["since"] == datetime(2026, 5, 1, tzinfo=timezone.utc)
    assert kwargs["until"] == datetime(2026, 5, 15, tzinfo=timezone.utc)
    assert kwargs["top_runs_limit"] == 5


def test_executions_cost_summary_rejects_out_of_range_limit(
    mocker: pytest_mock.MockFixture,
) -> None:
    """top_runs_limit must be within [1, 50]."""
    mock_fn = mocker.patch(
        "backend.api.features.executions.routes.get_user_cost_summary",
        AsyncMock(),
    )

    response = client.get("/executions/cost-summary?top_runs_limit=500")

    assert response.status_code == 422
    mock_fn.assert_not_awaited()


def test_executions_cost_summary_rejects_inverted_window(
    mocker: pytest_mock.MockFixture,
) -> None:
    """`since > until` is bad client input — surface 422, don't quietly return zeros."""
    mock_fn = mocker.patch(
        "backend.api.features.executions.routes.get_user_cost_summary",
        AsyncMock(),
    )

    response = client.get(
        "/executions/cost-summary"
        "?since=2026-05-15T00:00:00Z"
        "&until=2026-05-01T00:00:00Z"
    )

    assert response.status_code == 422
    mock_fn.assert_not_awaited()
