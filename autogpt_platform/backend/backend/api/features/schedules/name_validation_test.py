from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.features.schedules import routes
from backend.api.features.schedules.model import ScheduleCreationRequest
from backend.data.schedule import MAX_SCHEDULE_NAME_LENGTH


@pytest.mark.parametrize(
    "name", ["", "   ", "\t\n", "\u2003", "x" * (MAX_SCHEDULE_NAME_LENGTH + 1)]
)
def test_blank_or_overlong_schedule_name_is_rejected_before_dispatch(name: str):
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.requires_user] = lambda: None
    app.dependency_overrides[routes.get_user_id] = lambda: "user-1"
    app.dependency_overrides[routes.get_request_context] = lambda: Mock(
        org_id=None, team_id=None
    )
    with (
        patch.object(
            routes.graph_db, "get_graph", AsyncMock(return_value=None)
        ) as graph,
        patch.object(routes, "get_scheduler_client") as scheduler,
        TestClient(app) as client,
    ):
        response = client.post(
            "/graphs/graph-1/schedules",
            json={"name": name, "cron": "0 9 * * *", "inputs": {}},
        )
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "name"]
    graph.assert_not_awaited()
    scheduler.assert_not_called()


def test_schedule_name_is_trimmed_without_changing_internal_spaces():
    request = ScheduleCreationRequest(
        name="  Weekly  report ", cron="0 9 * * *", inputs={}
    )
    assert request.name == "Weekly  report"


def test_a_name_at_the_limit_is_trimmed_then_accepted():
    name = "x" * MAX_SCHEDULE_NAME_LENGTH
    request = ScheduleCreationRequest(name=f"  {name} ", cron="0 9 * * *", inputs={})
    assert request.name == name
