from datetime import datetime, timezone
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
from pytest_mock import MockerFixture

from .models import (
    HomeAction,
    HomeAttentionItem,
    HomeBriefing,
    HomeDashboardResponse,
    HomeTeamSummary,
    HomeWeekSummary,
)
from .routes import router

app = fastapi.FastAPI()
app.include_router(router)
client = fastapi.testclient.TestClient(app)

NOW = datetime(2026, 8, 10, 9, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    from autogpt_libs.auth import get_request_context
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    app.dependency_overrides[get_request_context] = mock_jwt_user["get_request_context"]
    yield
    app.dependency_overrides.clear()


def _dashboard() -> HomeDashboardResponse:
    return HomeDashboardResponse(
        generated_at=NOW,
        timezone="UTC",
        attention=[
            HomeAttentionItem(
                id="approval-node",
                kind="approval",
                priority="high",
                title="Send the prepared message",
                description="Your agent paused before taking an external action.",
                why_it_matters="The task cannot continue until you decide.",
                primary_action=HomeAction(label="Review", href="/library"),
            )
        ],
        briefing=HomeBriefing(
            generated_at=NOW,
            window_started_at=NOW,
            completed_count=0,
            failed_count=0,
            routine_count=0,
            outcomes=[],
        ),
        active_tasks=[],
        upcoming_tasks=[],
        team=HomeTeamSummary(total=0, ready=0, working=0, needs_attention=0),
        agents=[],
        week=HomeWeekSummary(
            run_count=0,
            completed_count=0,
            review_count=0,
            failed_count=0,
            total_runtime_seconds=0,
            timed_run_count=0,
            total_cost_cents=0,
            credits_balance=None,
            daily=[],
        ),
    )


def test_get_home_dashboard_returns_single_payload(
    mocker: MockerFixture, test_user_id: str
) -> None:
    build = mocker.patch(
        "backend.api.features.home.routes.build_home_dashboard",
        AsyncMock(return_value=_dashboard()),
    )
    mocker.patch(
        "backend.api.features.home.routes.get_user_team_ids",
        AsyncMock(return_value=["test-team"]),
    )

    response = client.get("/home")

    assert response.status_code == 200
    body = response.json()
    assert body["timezone"] == "UTC"
    assert body["attention"][0]["kind"] == "approval"
    assert set(body) >= {"attention", "briefing", "active_tasks", "team", "week"}
    build.assert_awaited_once_with(
        user_id=test_user_id, organization_id="test-org", team_ids=["test-team"]
    )


def test_personal_context_skips_team_lookup(
    mocker: MockerFixture, test_user_id: str
) -> None:
    from autogpt_libs.auth import get_request_context
    from autogpt_libs.auth.models import RequestContext

    app.dependency_overrides[get_request_context] = lambda: RequestContext(
        user_id=test_user_id,
        org_id=None,
        team_id=None,
        is_org_owner=False,
        is_org_admin=False,
        is_org_billing_manager=False,
        is_team_admin=False,
        is_team_billing_manager=False,
        seat_status=None,
    )
    build = mocker.patch(
        "backend.api.features.home.routes.build_home_dashboard",
        AsyncMock(return_value=_dashboard()),
    )
    get_team_ids = mocker.patch(
        "backend.api.features.home.routes.get_user_team_ids", AsyncMock(return_value=[])
    )

    assert client.get("/home").status_code == 200
    get_team_ids.assert_not_awaited()
    build.assert_awaited_once_with(
        user_id=test_user_id, organization_id=None, team_ids=[]
    )
