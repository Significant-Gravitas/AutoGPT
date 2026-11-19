"""Tests for the coming run times of the user's schedules."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from autogpt_libs.auth.dependencies import get_request_context
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from autogpt_libs.auth.models import RequestContext
from prisma.enums import TaskCreator, TaskStatus

from backend.api.features.schedules.upcoming_routes import coming_runs, router
from backend.data.task import Task
from backend.data.task_schedule_meta import TaskRepeat, TaskWatch
from backend.executor import scheduler

app = fastapi.FastAPI()
app.include_router(router)
client = fastapi.testclient.TestClient(app)

_MODULE = "backend.api.features.schedules.upcoming_routes"
_T0 = datetime(2026, 11, 18, 8, 30, tzinfo=UTC)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user, test_user_id):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]

    async def _context() -> RequestContext:
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

    app.dependency_overrides[get_request_context] = _context
    yield
    app.dependency_overrides.clear()


def _task(task_id: str, repeat: TaskRepeat | None) -> Task:
    return Task(
        id=task_id,
        user_id="test-user-id",
        title=task_id,
        status=TaskStatus.TODO,
        created_by=TaskCreator.USER,
        created_at=_T0,
        updated_at=_T0,
        repeat=repeat,
    )


def _workflow(schedule_id: str, cron: str) -> scheduler.GraphExecutionJobInfo:
    return scheduler.GraphExecutionJobInfo(
        id=schedule_id,
        name="Daily digest",
        next_run_time=_T0.isoformat(),
        user_id="test-user-id",
        graph_id="graph-1",
        graph_version=1,
        cron=cron,
        input_data={},
        timezone="UTC",
    )


def test_coming_runs_follow_the_cron_in_its_timezone():
    runs = coming_runs(["0 9 * * *"], "Europe/London", _T0, 3)
    assert runs == [
        datetime(2026, 11, 18, 9, 0, tzinfo=UTC),
        datetime(2026, 11, 19, 9, 0, tzinfo=UTC),
        datetime(2026, 11, 20, 9, 0, tzinfo=UTC),
    ]


def test_coming_runs_of_an_unreadable_cron_are_none():
    assert coming_runs(["not a cron"], "UTC", _T0, 3) == []


def test_lists_repeating_tasks_and_workflow_schedules_of_the_caller(
    mocker: pytest_mock.MockerFixture, test_user_id: str
):
    watch = TaskWatch(
        webhook_id="w", provider="github", trigger_block_id="b", line="PR opened"
    )
    list_tasks = mocker.patch(
        f"{_MODULE}.list_tasks",
        AsyncMock(
            return_value=[
                _task("daily", TaskRepeat(crons=["0 9 * * *"], label="Every day")),
                _task("once", None),
                _task("watching", TaskRepeat(label="On a new PR", watch=watch)),
            ]
        ),
    )
    workflows = mocker.patch(
        f"{_MODULE}.list_all_graphs_execution_schedules",
        AsyncMock(return_value=[_workflow("sched-1", "30 7 * * 1-5")]),
    )

    response = client.get("/schedules/upcoming", params={"count": 2})

    assert response.status_code == 200
    entries = response.json()["schedules"]
    assert [(e["kind"], e["id"], len(e["runs"])) for e in entries] == [
        ("task", "daily", 2),
        ("workflow", "sched-1", 2),
    ]
    assert list_tasks.await_args.kwargs["user_id"] == test_user_id
    assert workflows.await_args.kwargs["user_id"] == test_user_id


@pytest.mark.parametrize("count", [0, 25])
def test_refuses_a_count_out_of_range(count: int):
    response = client.get("/schedules/upcoming", params={"count": count})
    assert response.status_code == 422


def test_refuses_a_caller_without_a_login():
    app.dependency_overrides.pop(get_jwt_payload)
    assert client.get("/schedules/upcoming").status_code == 401
