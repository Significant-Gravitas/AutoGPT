"""Tests for the task-spine API routes.

Pattern mirrors backend/api/features/experts/routes_test.py: a local FastAPI
app, the global `mock_jwt_user` auth override fixture, and `tasks_db` mocked
with AsyncMock at the route module's import site.

The tenancy tests here assert that every handler *forwards the caller's own
user id* into the DB layer — that forwarding is what makes user A's tasks
unreachable to user B, and it is the thing a refactor is most likely to drop.
The end-to-end proof against a real database lives in ``tasks_db_test.py``.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from autogpt_libs.auth.dependencies import get_request_context
from autogpt_libs.auth.jwt_utils import get_jwt_payload

from backend.api.features.tasks.errors import DelegatedTaskNotFoundError
from backend.api.features.tasks.models import (
    DelegatedTask,
    DelegatedTaskDetail,
    TaskEvent,
    TaskExpertRef,
)
from backend.api.features.tasks.routes import router
from backend.util.exceptions import TaskDelegationRefusedError

app = fastapi.FastAPI()
app.include_router(router)

client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    """Setup auth overrides for all tests in this module"""
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    app.dependency_overrides[get_request_context] = mock_jwt_user["get_request_context"]
    yield
    app.dependency_overrides.clear()


def _make_task(**overrides) -> DelegatedTask:
    now = datetime(2026, 8, 30, 9, 0, tzinfo=timezone.utc)
    values = {
        "id": "task-1",
        "title": "Draft the weekly report",
        "spec": "Run Weekly Report with:\n- week: current",
        "status": "WORKING",
        "acceptance": "PENDING",
        "created_by_type": "USER",
        "created_by_id": "user-1",
        "owner": TaskExpertRef(
            id="expert-maria",
            name="Maria",
            avatar_url=None,
            role="Marketing Strategist",
        ),
        "parent_task_id": None,
        "root_task_id": "task-1",
        "origin_session_id": "session-1",
        "ancestor_expert_ids": ["expert-maria"],
        "handoff_count": 0,
        "revision_count": 0,
        "spend_total": 250,
        "outcome_summary": None,
        "amendments": [],
        "created_at": now,
        "updated_at": now,
        "runs": [],
    }
    values.update(overrides)
    return DelegatedTask(**values)


# ─── list ──────────────────────────────────────────────────────────────


def test_list_tasks_returns_the_callers_tasks(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    mock_list = mocker.patch(
        "backend.api.features.tasks.routes.tasks_db.list_tasks",
        new_callable=AsyncMock,
        return_value=[_make_task()],
    )

    response = client.get("/tasks")

    assert response.status_code == 200
    data = response.json()
    assert [task["id"] for task in data] == ["task-1"]
    assert data[0]["owner"]["name"] == "Maria"
    mock_list.assert_awaited_once_with(
        test_user_id, expert_id=None, status=None, limit=50
    )


def test_list_tasks_forwards_expert_and_status_filters(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    mock_list = mocker.patch(
        "backend.api.features.tasks.routes.tasks_db.list_tasks",
        new_callable=AsyncMock,
        return_value=[],
    )

    response = client.get("/tasks?expert_id=expert-maria&status=DONE&limit=5")

    assert response.status_code == 200
    mock_list.assert_awaited_once_with(
        test_user_id, expert_id="expert-maria", status="DONE", limit=5
    )


def test_list_tasks_rejects_an_unknown_status() -> None:
    response = client.get("/tasks?status=NOPE")

    assert response.status_code == 422


def test_list_tasks_rejects_an_oversized_limit() -> None:
    """The cap exists so one user's history can't turn into an unbounded
    scan; a client asking past it must be refused, not silently clamped."""
    response = client.get("/tasks?limit=500")

    assert response.status_code == 422


# ─── events ────────────────────────────────────────────────────────────


def test_list_task_events_returns_the_callers_events_in_the_fe_shape(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    """The exact snake_case shape is the office view's polling contract."""
    mock_events = mocker.patch(
        "backend.api.features.tasks.routes.tasks_db.list_task_events",
        new_callable=AsyncMock,
        return_value=[
            TaskEvent(
                task_id="task-1",
                expert_id="expert-maria",
                event="working",
                ts="2026-08-30T09:00:00+00:00",
            )
        ],
    )

    response = client.get("/tasks/events")

    assert response.status_code == 200
    assert response.json() == {
        "events": [
            {
                "task_id": "task-1",
                "expert_id": "expert-maria",
                "event": "working",
                "ts": "2026-08-30T09:00:00+00:00",
            }
        ]
    }
    mock_events.assert_awaited_once_with(test_user_id, since=None)


def test_list_task_events_forwards_since_as_a_datetime(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    mock_events = mocker.patch(
        "backend.api.features.tasks.routes.tasks_db.list_task_events",
        new_callable=AsyncMock,
        return_value=[],
    )

    response = client.get("/tasks/events?since=2026-08-30T09:00:00Z")

    assert response.status_code == 200
    assert response.json() == {"events": []}
    mock_events.assert_awaited_once_with(
        test_user_id, since=datetime(2026, 8, 30, 9, 0, tzinfo=timezone.utc)
    )


def test_list_task_events_rejects_a_malformed_since() -> None:
    response = client.get("/tasks/events?since=not-a-date")

    assert response.status_code == 422


# ─── detail ────────────────────────────────────────────────────────────


def test_get_task_returns_the_task_and_its_children(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    mock_get = mocker.patch(
        "backend.api.features.tasks.routes.tasks_db.get_task",
        new_callable=AsyncMock,
        return_value=DelegatedTaskDetail(
            task=_make_task(),
            children=[_make_task(id="task-2", parent_task_id="task-1")],
        ),
    )

    response = client.get("/tasks/task-1")

    assert response.status_code == 200
    data = response.json()
    assert data["task"]["id"] == "task-1"
    assert [child["id"] for child in data["children"]] == ["task-2"]
    mock_get.assert_awaited_once_with(test_user_id, "task-1")


def test_get_task_returns_404_for_another_users_task(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    """``get_task`` answers None for both "missing" and "not yours", and the
    route must not distinguish them — otherwise a 404-vs-403 split would
    confirm the existence of another user's task id."""
    mock_get = mocker.patch(
        "backend.api.features.tasks.routes.tasks_db.get_task",
        new_callable=AsyncMock,
        return_value=None,
    )

    response = client.get("/tasks/someone-elses-task")

    assert response.status_code == 404
    assert response.json()["detail"] == "Task not found"
    mock_get.assert_awaited_once_with(test_user_id, "someone-elses-task")


# ─── cancel ────────────────────────────────────────────────────────────


def test_cancel_task_cancels_and_returns_the_tree(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    mock_cancel = mocker.patch(
        "backend.api.features.tasks.routes.tasks_db.cancel_task",
        new_callable=AsyncMock,
        return_value=DelegatedTaskDetail(
            task=_make_task(status="CANCELLED"),
            children=[_make_task(id="task-2", status="CANCELLED")],
        ),
    )

    response = client.post("/tasks/task-1/cancel")

    assert response.status_code == 200
    data = response.json()
    assert data["task"]["status"] == "CANCELLED"
    assert data["children"][0]["status"] == "CANCELLED"
    mock_cancel.assert_awaited_once_with(test_user_id, "task-1")


def test_cancel_task_returns_404_for_another_users_task(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    mock_cancel = mocker.patch(
        "backend.api.features.tasks.routes.tasks_db.cancel_task",
        new_callable=AsyncMock,
        side_effect=DelegatedTaskNotFoundError("someone-elses-task"),
    )

    response = client.post("/tasks/someone-elses-task/cancel")

    assert response.status_code == 404
    mock_cancel.assert_awaited_once_with(test_user_id, "someone-elses-task")


# ─── answer ────────────────────────────────────────────────────────────


def test_answer_task_resumes_the_worker_session(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    task = _make_task(status="WORKING")
    mock_answer = mocker.patch(
        "backend.api.features.tasks.routes.task_actions.answer_delegated_task",
        new_callable=AsyncMock,
        return_value=(task, "worker-session-1"),
    )
    mock_queue = mocker.patch(
        "backend.api.features.tasks.routes.queue_user_message",
        new_callable=AsyncMock,
        return_value=mocker.Mock(turn_in_flight=False),
    )
    mock_schedule = mocker.patch(
        "backend.api.features.tasks.routes.schedule_chat_turn",
        new_callable=AsyncMock,
        return_value="turn-1",
    )

    response = client.post("/tasks/task-1/answer", json={"answer": "Staging"})

    assert response.status_code == 200
    assert response.json()["status"] == "WORKING"
    mock_answer.assert_awaited_once_with(test_user_id, "task-1", answer="Staging")
    assert mock_queue.await_args.kwargs["session_id"] == "worker-session-1"
    schedule_kwargs = mock_schedule.await_args.kwargs
    assert schedule_kwargs["session_id"] == "worker-session-1"
    assert schedule_kwargs["user_id"] == test_user_id
    assert "Staging" in schedule_kwargs["message"]


def test_answer_task_injects_into_a_running_turn_without_scheduling(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.api.features.tasks.routes.task_actions.answer_delegated_task",
        new_callable=AsyncMock,
        return_value=(_make_task(status="WORKING"), "worker-session-1"),
    )
    mocker.patch(
        "backend.api.features.tasks.routes.queue_user_message",
        new_callable=AsyncMock,
        return_value=mocker.Mock(turn_in_flight=True),
    )
    mock_schedule = mocker.patch(
        "backend.api.features.tasks.routes.schedule_chat_turn",
        new_callable=AsyncMock,
    )

    response = client.post("/tasks/task-1/answer", json={"answer": "Prod"})

    assert response.status_code == 200
    mock_schedule.assert_not_awaited()


def test_answer_task_404s_when_the_task_is_missing(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.api.features.tasks.routes.task_actions.answer_delegated_task",
        new_callable=AsyncMock,
        side_effect=DelegatedTaskNotFoundError("task-9"),
    )

    response = client.post("/tasks/task-9/answer", json={"answer": "Hi"})

    assert response.status_code == 404


def test_answer_task_409s_when_the_task_is_not_waiting(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.api.features.tasks.routes.task_actions.answer_delegated_task",
        new_callable=AsyncMock,
        side_effect=TaskDelegationRefusedError("This task is not waiting."),
    )

    response = client.post("/tasks/task-1/answer", json={"answer": "Hi"})

    assert response.status_code == 409


def test_answer_task_rejects_a_blank_answer() -> None:
    response = client.post("/tasks/task-1/answer", json={"answer": "   "})

    assert response.status_code == 422


# ─── accept / reject ───────────────────────────────────────────────────


def test_accept_task_forwards_the_callers_user_id(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    mock_accept = mocker.patch(
        "backend.api.features.tasks.routes.task_review.accept_delegated_task",
        new_callable=AsyncMock,
        return_value=_make_task(status="DONE", acceptance="ACCEPTED"),
    )

    response = client.post("/tasks/task-1/accept")

    assert response.status_code == 200
    data = response.json()
    assert data["task"]["acceptance"] == "ACCEPTED"
    assert data["escalated"] is False
    mock_accept.assert_awaited_once_with(test_user_id, "task-1")


def test_accept_task_409s_when_the_task_is_open(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.api.features.tasks.routes.task_review.accept_delegated_task",
        new_callable=AsyncMock,
        side_effect=TaskDelegationRefusedError("Only a finished task…"),
    )

    response = client.post("/tasks/task-1/accept")

    assert response.status_code == 409


def test_reject_task_reopens_it_and_nudges_the_owner_session(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    reopened = _make_task(
        status="WORKING",
        acceptance="PENDING",
        revision_count=1,
        origin_session_id="worker-session-1",
        outcome_summary="First draft used Q2 numbers.",
    )
    mock_reject = mocker.patch(
        "backend.api.features.tasks.routes.task_review.reject_delegated_task",
        new_callable=AsyncMock,
        return_value=(reopened, True),
    )
    mocker.patch(
        "backend.api.features.tasks.routes.queue_user_message",
        new_callable=AsyncMock,
        return_value=mocker.Mock(turn_in_flight=False),
    )
    mock_schedule = mocker.patch(
        "backend.api.features.tasks.routes.schedule_chat_turn",
        new_callable=AsyncMock,
    )

    response = client.post("/tasks/task-1/reject", json={"note": "Use Q3 numbers"})

    assert response.status_code == 200
    data = response.json()
    assert data["escalated"] is False
    assert data["task"]["id"] == "task-1"
    assert data["task"]["status"] == "WORKING"
    mock_reject.assert_awaited_once_with(test_user_id, "task-1", note="Use Q3 numbers")
    schedule_kwargs = mock_schedule.await_args.kwargs
    assert schedule_kwargs["session_id"] == "worker-session-1"
    assert "Use Q3 numbers" in schedule_kwargs["message"]
    assert "report_task" in schedule_kwargs["message"]


def test_reject_task_at_the_cap_escalates_without_scheduling(
    mocker: pytest_mock.MockerFixture,
) -> None:
    parent = _make_task(status="DONE", acceptance="REJECTED", revision_count=2)
    mocker.patch(
        "backend.api.features.tasks.routes.task_review.reject_delegated_task",
        new_callable=AsyncMock,
        return_value=(parent, False),
    )
    mock_schedule = mocker.patch(
        "backend.api.features.tasks.routes.schedule_chat_turn",
        new_callable=AsyncMock,
    )

    response = client.post("/tasks/task-1/reject", json={"note": "Still wrong"})

    assert response.status_code == 200
    data = response.json()
    assert data["escalated"] is True
    assert data["revision_task"] is None
    assert "clarify" in data["message"]
    mock_schedule.assert_not_awaited()


def test_reject_task_rejects_a_blank_note() -> None:
    response = client.post("/tasks/task-1/reject", json={"note": "  "})

    assert response.status_code == 422
