"""Tests for the execution routes."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from fastapi.routing import APIRoute

from backend.api.features.executions.routes import router
from backend.api.rest_api import app as real_app
from backend.data import execution as execution_db
from backend.util.exceptions import NotFoundError
from backend.util.models import Pagination

app = fastapi.FastAPI()
app.include_router(router)
client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user, test_user_id):
    from autogpt_libs.auth.dependencies import get_request_context
    from autogpt_libs.auth.jwt_utils import get_jwt_payload
    from autogpt_libs.auth.models import RequestContext

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]

    # The real get_request_context queries Prisma to resolve the personal org,
    # which closes the test event loop between sync TestClient calls.
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


def _paginated(*executions):
    return execution_db.GraphExecutionsPaginated(
        executions=list(executions),
        pagination=Pagination(
            total_items=len(executions), total_pages=1, current_page=1, page_size=25
        ),
    )


def test_list_all_executions_applies_the_activity_gate(
    mocker: pytest_mock.MockFixture,
) -> None:
    """The gate is what keeps flag-disabled activity summaries out of the
    response; bypassing it leaks them to every caller."""
    mocker.patch(
        "backend.api.features.executions.routes.execution_db"
        ".get_graph_executions_paginated",
        AsyncMock(return_value=_paginated()),
    )
    gate = mocker.patch(
        "backend.api.features.executions.routes.hide_activity_summaries_if_disabled",
        AsyncMock(return_value=[]),
    )

    response = client.get("/executions")

    assert response.status_code == 200
    gate.assert_awaited_once()


def test_list_all_executions_is_not_scoped_to_a_graph(
    mocker: pytest_mock.MockFixture,
) -> None:
    """This route and the per-graph one share a db call; a graph_id here would
    silently narrow the user-wide list."""
    query = mocker.patch(
        "backend.api.features.executions.routes.execution_db"
        ".get_graph_executions_paginated",
        AsyncMock(return_value=_paginated()),
    )
    mocker.patch(
        "backend.api.features.executions.routes.hide_activity_summaries_if_disabled",
        AsyncMock(return_value=[]),
    )

    client.get("/executions")

    assert "graph_id" not in query.await_args.kwargs


def test_list_graph_executions_forwards_pagination(
    mocker: pytest_mock.MockFixture,
) -> None:
    query = mocker.patch(
        "backend.api.features.executions.routes.execution_db"
        ".get_graph_executions_paginated",
        AsyncMock(return_value=_paginated()),
    )
    mocker.patch(
        "backend.api.features.executions.routes.hide_activity_summaries_if_disabled",
        AsyncMock(return_value=[]),
    )
    mocker.patch(
        "backend.api.features.executions.routes.get_user_onboarding",
        AsyncMock(return_value=Mock(onboardingAgentExecutionId=None)),
    )

    response = client.get("/graphs/graph-1/executions?page=3&page_size=10")

    assert response.status_code == 200
    kwargs = query.await_args.kwargs
    assert (kwargs["graph_id"], kwargs["page"], kwargs["page_size"]) == (
        "graph-1",
        3,
        10,
    )


def test_delete_execution_returns_204(mocker: pytest_mock.MockFixture) -> None:
    deleted = mocker.patch(
        "backend.api.features.executions.routes.execution_db.delete_graph_execution",
        AsyncMock(),
    )

    response = client.delete("/executions/exec-1")

    assert response.status_code == 204
    assert deleted.await_args.kwargs["graph_exec_id"] == "exec-1"


def test_get_shared_execution_returns_404_for_an_unknown_token(
    mocker: pytest_mock.MockFixture,
) -> None:
    mocker.patch(
        "backend.api.features.executions.routes.execution_db"
        ".get_graph_execution_by_share_token",
        AsyncMock(return_value=None),
    )

    response = client.get("/public/shared/550e8400-e29b-41d4-a716-446655440000")

    assert response.status_code == 404


def test_get_shared_execution_rejects_a_malformed_token() -> None:
    """The token is the only credential this public route has, so its pattern
    is the access control."""
    response = client.get("/public/shared/not-a-uuid")

    assert response.status_code == 422


def test_get_graph_execution_rejects_a_graph_id_mismatch(
    mocker: pytest_mock.MockFixture,
) -> None:
    """The execution is fetched by id alone, so the graph_id in the path is
    only checked here — dropping it would let any graph's URL read any of the
    caller's executions."""
    mocker.patch(
        "backend.api.features.executions.routes.execution_db.get_graph_execution",
        AsyncMock(return_value=Mock(graph_id="other-graph", graph_version=1)),
    )

    response = client.get("/graphs/graph-1/executions/exec-1")

    assert response.status_code == 404


def test_enable_sharing_clears_stale_allowlist_before_issuing_a_token(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Order matters: leaving old file records in place while a new token is
    written would expose files the previous share allowed."""
    calls: list[str] = []
    mocker.patch(
        "backend.api.features.executions.routes.execution_db.get_graph_execution",
        AsyncMock(return_value=Mock(outputs={})),
    )
    mocker.patch(
        "backend.api.features.executions.routes.execution_db"
        ".delete_shared_execution_files",
        AsyncMock(side_effect=lambda **_: calls.append("delete")),
    )
    mocker.patch(
        "backend.api.features.executions.routes.execution_db"
        ".update_graph_execution_share_status",
        AsyncMock(side_effect=lambda **_: calls.append("update")),
    )
    mocker.patch(
        "backend.api.features.executions.routes.execution_db"
        ".create_shared_execution_files",
        AsyncMock(side_effect=lambda **_: calls.append("create")),
    )

    response = client.post("/graphs/graph-1/executions/exec-1/share")

    assert response.status_code == 200
    assert calls == ["delete", "update", "create"]
    assert response.json()["share_url"].endswith(response.json()["share_token"])


def test_enable_sharing_maps_a_lost_execution_to_404(
    mocker: pytest_mock.MockFixture,
) -> None:
    """The write enforces (id, user_id) at the DB layer, so a delete racing the
    pre-check must surface as 404 rather than a silent no-op."""
    mocker.patch(
        "backend.api.features.executions.routes.execution_db.get_graph_execution",
        AsyncMock(return_value=Mock(outputs={})),
    )
    mocker.patch(
        "backend.api.features.executions.routes.execution_db"
        ".delete_shared_execution_files",
        AsyncMock(),
    )
    mocker.patch(
        "backend.api.features.executions.routes.execution_db"
        ".update_graph_execution_share_status",
        AsyncMock(side_effect=NotFoundError("gone")),
    )

    response = client.post("/graphs/graph-1/executions/exec-1/share")

    assert response.status_code == 404


def test_enable_sharing_requires_an_execution(
    mocker: pytest_mock.MockFixture,
) -> None:
    mocker.patch(
        "backend.api.features.executions.routes.execution_db.get_graph_execution",
        AsyncMock(return_value=None),
    )

    response = client.post("/graphs/graph-1/executions/exec-1/share")

    assert response.status_code == 404


def test_disable_sharing_revokes_the_token_and_the_file_allowlist(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Revoking the token without clearing the allowlist would leave the files
    reachable to anyone who kept the old link."""
    mocker.patch(
        "backend.api.features.executions.routes.execution_db.get_graph_execution",
        AsyncMock(return_value=Mock()),
    )
    delete_files = mocker.patch(
        "backend.api.features.executions.routes.execution_db"
        ".delete_shared_execution_files",
        AsyncMock(),
    )
    update = mocker.patch(
        "backend.api.features.executions.routes.execution_db"
        ".update_graph_execution_share_status",
        AsyncMock(),
    )

    response = client.delete("/graphs/graph-1/executions/exec-1/share")

    assert response.status_code in (200, 204)
    delete_files.assert_awaited_once()
    assert update.await_args.kwargs["is_shared"] is False
