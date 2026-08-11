from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture

from backend.data.execution import ExecutionStatus, GraphExecutionMeta
from backend.data.execution_cost_summary import UserExecutionCostSummary

from .service import build_home_dashboard


def _execution() -> GraphExecutionMeta:
    # `build_home_dashboard` reads the live clock, so anchor the run to now — a
    # fixed timestamp would drop out of the 24h briefing window over time.
    ran_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    return GraphExecutionMeta(
        id="exec-1",
        user_id="user-1",
        graph_id="graph-1",
        graph_version=1,
        inputs=None,
        credential_inputs=None,
        nodes_input_masks=None,
        preset_id=None,
        status=ExecutionStatus.COMPLETED,
        started_at=ran_at,
        ended_at=ran_at,
        stats=GraphExecutionMeta.Stats(activity_status="Booked the flight."),
    )


def _cost_summary() -> UserExecutionCostSummary:
    return UserExecutionCostSummary(
        total_cents=0,
        run_count=0,
        billable_run_count=0,
        failed_cost_cents=0,
        by_agent=[],
        top_runs=[],
        daily=[],
    )


@pytest.fixture
def home_dependencies(mocker: MockerFixture):
    mocker.patch(
        "backend.api.features.home.service.experts_db.list_experts",
        AsyncMock(return_value=[]),
    )
    mocker.patch(
        "backend.api.features.home.service.execution_db.get_graph_executions",
        AsyncMock(return_value=[_execution()]),
    )
    mocker.patch(
        "backend.api.features.home.service.review_db.get_pending_reviews_for_user",
        AsyncMock(return_value=[]),
    )
    mocker.patch(
        "backend.api.features.home.service.get_user_cost_summary",
        AsyncMock(return_value=_cost_summary()),
    )
    mocker.patch(
        "backend.api.features.home.service.library_db.get_library_agent_refs_by_graph_ids",
        AsyncMock(return_value=[]),
    )
    mocker.patch(
        "backend.api.features.home.service.user_db.get_user_by_id",
        AsyncMock(return_value=None),
    )
    scheduler = MagicMock()
    scheduler.get_execution_schedules = AsyncMock(return_value=[])
    mocker.patch(
        "backend.api.features.home.service.get_scheduler_client", return_value=scheduler
    )
    credit_model = MagicMock()
    credit_model.get_credits = AsyncMock(return_value=100)
    mocker.patch(
        "backend.api.features.home.service.get_credit_model",
        AsyncMock(return_value=credit_model),
    )


@pytest.mark.asyncio
async def test_activity_summary_surfaces_when_flag_enabled(
    mocker: MockerFixture, home_dependencies
) -> None:
    mocker.patch(
        "backend.api.features.executions.activity_gate.is_feature_enabled",
        AsyncMock(return_value=True),
    )

    dashboard = await build_home_dashboard(user_id="user-1")

    assert dashboard.briefing.outcomes[0].title == "Booked the flight."


@pytest.mark.asyncio
async def test_activity_summary_hidden_when_flag_disabled(
    mocker: MockerFixture, home_dependencies
) -> None:
    mocker.patch(
        "backend.api.features.executions.activity_gate.is_feature_enabled",
        AsyncMock(return_value=False),
    )

    dashboard = await build_home_dashboard(user_id="user-1")

    assert dashboard.briefing.outcomes[0].title == "Agent task finished"


@pytest.mark.asyncio
async def test_schedules_stay_owner_scoped_inside_an_organization(
    mocker: MockerFixture, home_dependencies
) -> None:
    mocker.patch(
        "backend.api.features.executions.activity_gate.is_feature_enabled",
        AsyncMock(return_value=True),
    )
    scheduler = MagicMock()
    scheduler.get_execution_schedules = AsyncMock(return_value=[])
    mocker.patch(
        "backend.api.features.home.service.get_scheduler_client", return_value=scheduler
    )
    credit_model = MagicMock()
    credit_model.get_credits = AsyncMock(return_value=100)
    get_credit_model = mocker.patch(
        "backend.api.features.home.service.get_credit_model",
        AsyncMock(return_value=credit_model),
    )

    await build_home_dashboard(user_id="user-1", organization_id="org-1")

    # Executions, reviews and cost totals are personal, so a teammate's schedule
    # would show an upcoming run whose outcome never lands anywhere else.
    scheduler.get_execution_schedules.assert_awaited_once_with(user_id="user-1")
    get_credit_model.assert_awaited_once_with("user-1", "org-1")


@pytest.mark.asyncio
async def test_scheduler_and_credit_failures_degrade_instead_of_failing_the_page(
    mocker: MockerFixture, home_dependencies
) -> None:
    mocker.patch(
        "backend.api.features.executions.activity_gate.is_feature_enabled",
        AsyncMock(return_value=True),
    )
    scheduler = MagicMock()
    scheduler.get_execution_schedules = AsyncMock(side_effect=RuntimeError("scheduler"))
    mocker.patch(
        "backend.api.features.home.service.get_scheduler_client", return_value=scheduler
    )
    mocker.patch(
        "backend.api.features.home.service.get_credit_model",
        AsyncMock(side_effect=RuntimeError("credits")),
    )

    dashboard = await build_home_dashboard(user_id="user-1")

    assert dashboard.upcoming_tasks == []
    assert dashboard.week.credits_balance is None
    assert dashboard.briefing.outcomes[0].title == "Booked the flight."
