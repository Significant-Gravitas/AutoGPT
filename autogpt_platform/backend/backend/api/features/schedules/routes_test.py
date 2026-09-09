"""Tests for the schedule routes."""

from unittest.mock import AsyncMock, Mock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from fastapi.routing import APIRoute

from backend.api.features.schedules.routes import router
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


EXPECTED_OPERATIONS = {
    ("post", "/api/graphs/{graph_id}/schedules"): "postV1Create execution schedule",
    ("get", "/api/graphs/{graph_id}/schedules"): (
        "getV1List execution schedules for a graph"
    ),
    ("get", "/api/schedules"): "getV1List execution schedules for a user",
    ("get", "/api/schedules/followups"): "listCopilotFollowupSchedules",
    ("delete", "/api/schedules/{schedule_id}"): "deleteV1Delete execution schedule",
}


@pytest.mark.parametrize(
    "method,path,operation_id",
    [(m, p, oid) for (m, p), oid in EXPECTED_OPERATIONS.items()],
)
def test_schedule_operation_is_published(method: str, path: str, operation_id: str):
    """The mounted surface is contract: the generated frontend client is built from it."""
    operation = real_app.openapi()["paths"][path][method]
    assert operation["operationId"] == operation_id
    assert operation["tags"] == ["v1", "schedules"]


def test_schedule_surface_has_no_other_operations():
    published = {
        (method, path)
        for path, operations in real_app.openapi()["paths"].items()
        for method, operation in operations.items()
        if "schedules" in operation["tags"] and "v1" in operation["tags"]
    }
    assert published == set(EXPECTED_OPERATIONS)


@pytest.mark.parametrize("path", sorted({p for _, p in EXPECTED_OPERATIONS}))
def test_schedule_path_is_served_by_this_module(path: str):
    handlers = {
        route.endpoint.__module__
        for route in real_app.routes
        if isinstance(route, APIRoute) and route.path == path
    }
    assert handlers == {"backend.api.features.schedules.routes"}


def test_followups_is_not_shadowed_by_the_schedule_id_route():
    """Only registration order keeps these apart; a GET on {schedule_id} would win."""
    paths = [
        route.path
        for route in real_app.routes
        if isinstance(route, APIRoute) and route.path.startswith("/api/schedules/")
    ]
    assert paths.index("/api/schedules/followups") < paths.index(
        "/api/schedules/{schedule_id}"
    )


def test_list_copilot_turn_schedules_filters_to_copilot_kind(
    mocker: pytest_mock.MockFixture,
    test_user_id: str,
) -> None:
    """GET /schedules/followups returns only CopilotTurnJobInfo items for the user.

    The route delegates to ``Scheduler.get_execution_schedules(kind="copilot_turn")``;
    we mock the client to make sure (a) the kind filter is forwarded and
    (b) any non-copilot rows are dropped from the response.
    """
    from backend.executor.scheduler import CopilotTurnJobInfo, GraphExecutionJobInfo

    copilot_info = CopilotTurnJobInfo(
        id="sched-1",
        name="copilot followup",
        next_run_time="2026-05-22T10:00:00+00:00",
        timezone="UTC",
        user_id=test_user_id,
        session_id="sess-1",
        message="check status",
        cron="0 9 * * *",
    )
    graph_info = GraphExecutionJobInfo(
        id="sched-2",
        name="graph run",
        next_run_time="2026-05-22T11:00:00+00:00",
        timezone="UTC",
        user_id=test_user_id,
        graph_id="g-1",
        graph_version=1,
        cron="0 10 * * *",
        input_data={},
    )

    mock_client = Mock()
    mock_client.get_execution_schedules = AsyncMock(
        return_value=[copilot_info, graph_info]
    )
    mocker.patch(
        "backend.api.features.schedules.routes.get_scheduler_client",
        return_value=mock_client,
    )

    response = client.get("/schedules/followups")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == "sched-1"
    assert body[0]["kind"] == "copilot_turn"
    assert body[0]["session_id"] == "sess-1"

    mock_client.get_execution_schedules.assert_awaited_once_with(
        user_id=test_user_id, kind="copilot_turn"
    )
