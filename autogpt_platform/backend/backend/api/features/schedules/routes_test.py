"""Tests for the schedule routes."""

from unittest.mock import AsyncMock, Mock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from fastapi.routing import APIRoute

from backend.api.features.schedules.routes import router
from backend.api.rest_api import app as real_app
from backend.executor import scheduler
from backend.util.exceptions import NotFoundError

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


def _job_info(**overrides) -> scheduler.GraphExecutionJobInfo:
    defaults = {
        "id": "sched-1",
        "name": "nightly",
        "next_run_time": "2026-01-01T03:00:00+00:00",
        "user_id": "user-1",
        "graph_id": "graph-1",
        "graph_version": 3,
        "cron": "0 3 * * *",
        "input_data": {},
    }
    defaults.update(overrides)
    return scheduler.GraphExecutionJobInfo(**defaults)


def _patch_scheduler(mocker: pytest_mock.MockFixture, **methods) -> Mock:
    scheduler_client = Mock()
    for name, value in methods.items():
        setattr(scheduler_client, name, value)
    mocker.patch(
        "backend.api.features.schedules.routes.get_scheduler_client",
        return_value=scheduler_client,
    )
    return scheduler_client


def test_list_graph_execution_schedules_scopes_to_the_graph(
    mocker: pytest_mock.MockFixture,
) -> None:
    sched = _patch_scheduler(
        mocker, get_graph_execution_schedules=AsyncMock(return_value=[_job_info()])
    )
    mocker.patch(
        "backend.api.features.schedules.routes.get_user_team_ids",
        AsyncMock(return_value=["team-9"]),
    )

    response = client.get("/graphs/graph-1/schedules")

    assert response.status_code == 200
    assert [s["id"] for s in response.json()] == ["sched-1"]
    assert sched.get_graph_execution_schedules.await_args.kwargs["graph_id"] == (
        "graph-1"
    )


def test_list_all_schedules_does_not_scope_to_a_graph(
    mocker: pytest_mock.MockFixture,
) -> None:
    """The user-wide route shares a scheduler method with the per-graph one, so
    passing a graph_id here would silently narrow it."""
    sched = _patch_scheduler(
        mocker, get_graph_execution_schedules=AsyncMock(return_value=[])
    )
    mocker.patch(
        "backend.api.features.schedules.routes.get_user_team_ids",
        AsyncMock(return_value=[]),
    )

    response = client.get("/schedules")

    assert response.status_code == 200
    assert "graph_id" not in sched.get_graph_execution_schedules.await_args.kwargs


def test_delete_schedule_returns_the_id(mocker: pytest_mock.MockFixture) -> None:
    sched = _patch_scheduler(mocker, delete_schedule=AsyncMock())

    response = client.delete("/schedules/sched-1")

    assert response.status_code == 200
    assert response.json() == {"id": "sched-1"}
    assert sched.delete_schedule.await_args.args[0] == "sched-1"


def test_delete_schedule_maps_not_found_to_404(
    mocker: pytest_mock.MockFixture,
) -> None:
    _patch_scheduler(
        mocker, delete_schedule=AsyncMock(side_effect=NotFoundError("gone"))
    )

    response = client.delete("/schedules/missing")

    assert response.status_code == 404


def test_create_schedule_returns_404_for_an_unknown_graph(
    mocker: pytest_mock.MockFixture,
) -> None:
    mocker.patch(
        "backend.api.features.schedules.routes.graph_db.get_graph",
        AsyncMock(return_value=None),
    )

    response = client.post(
        "/graphs/graph-1/schedules",
        json={"name": "nightly", "cron": "0 3 * * *", "inputs": {}},
    )

    assert response.status_code == 404


def test_create_schedule_resolves_the_expert_when_none_is_given(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Attribution for schedules made through the generic UI rests on this
    lookup; without it every such schedule loses its expert."""
    mocker.patch(
        "backend.api.features.schedules.routes.graph_db.get_graph",
        AsyncMock(return_value=Mock(version=3)),
    )
    mocker.patch(
        "backend.api.features.schedules.routes.experts_db.resolve_expert_for_graph",
        AsyncMock(return_value="expert-7"),
    )
    sched = _patch_scheduler(
        mocker, add_execution_schedule=AsyncMock(return_value=_job_info())
    )
    mocker.patch(
        "backend.api.features.schedules.routes.complete_onboarding_step", AsyncMock()
    )

    response = client.post(
        "/graphs/graph-1/schedules",
        json={
            "name": "nightly",
            "cron": "0 3 * * *",
            "inputs": {},
            "timezone": "Europe/Amsterdam",
        },
    )

    assert response.status_code == 200
    kwargs = sched.add_execution_schedule.await_args.kwargs
    assert kwargs["expert_id"] == "expert-7"
    assert kwargs["user_timezone"] == "Europe/Amsterdam"


def test_create_schedule_rejects_an_archived_expert(
    mocker: pytest_mock.MockFixture,
) -> None:
    """An explicit expert_id must be live and the caller's; an archived one
    would otherwise attribute the schedule's spend to a retired expert."""
    mocker.patch(
        "backend.api.features.schedules.routes.graph_db.get_graph",
        AsyncMock(return_value=Mock(version=3)),
    )
    mocker.patch(
        "backend.api.features.schedules.routes.experts_db.get_expert",
        AsyncMock(return_value=Mock(is_archived=True)),
    )

    response = client.post(
        "/graphs/graph-1/schedules",
        json={
            "name": "nightly",
            "cron": "0 3 * * *",
            "inputs": {},
            "timezone": "UTC",
            "expert_id": "expert-7",
        },
    )

    assert response.status_code == 404
