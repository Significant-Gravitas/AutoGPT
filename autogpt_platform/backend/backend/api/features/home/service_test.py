from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture

from backend.copilot.briefing.models import BriefingContent, BriefingRunItem
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
    mocker.patch(
        "backend.api.features.home.service.briefing_db.get_briefing_for_date",
        AsyncMock(return_value=None),
    )


def _stored_briefing(**overrides) -> BriefingContent:
    item = BriefingRunItem(
        expert_id=None,
        expert_name=None,
        expert_avatar_url=None,
        agent_name="Inbox triage",
        graph_id="graph-1",
        execution_id="stored-run",
        library_agent_id=None,
        status="COMPLETED",
        summary="Filed the report. Nothing else was pending.",
        title="Filed the report.",
        detail="Nothing else was pending.",
        occurred_at=datetime.now(timezone.utc) - timedelta(hours=1),
        link=None,
    )
    return BriefingContent(
        generated_at=datetime.now(timezone.utc) - timedelta(minutes=30),
        timezone="UTC",
        zero_expert_fallback=True,
        run_items=[item],
        decision_items=[],
        decision_total=0,
        **overrides,
    )


def _patch_stored_briefing(mocker: MockerFixture, content: object) -> None:
    mocker.patch(
        "backend.api.features.home.service.briefing_db.get_briefing_for_date",
        AsyncMock(return_value=MagicMock(id="briefing-1", content=content)),
    )


@pytest.mark.asyncio
async def test_briefing_anchors_on_todays_persisted_row(
    mocker: MockerFixture, home_dependencies
) -> None:
    mocker.patch(
        "backend.api.features.executions.activity_gate.is_feature_enabled",
        AsyncMock(return_value=True),
    )
    mocker.patch(
        "backend.api.features.home.service.is_feature_enabled",
        AsyncMock(return_value=True),
    )
    _patch_stored_briefing(mocker, _stored_briefing().model_dump(mode="json"))

    dashboard = await build_home_dashboard(user_id="user-1")

    assert dashboard.briefing.source == "persisted"
    # The stored row anchors the card; the run that finished after it is
    # appended live rather than held back until tomorrow morning.
    assert [outcome.id for outcome in dashboard.briefing.outcomes] == [
        "stored-run",
        "exec-1",
    ]
    assert dashboard.briefing.outcomes[0].title == "Filed the report."
    assert dashboard.briefing.outcomes[1].title == "Booked the flight."


@pytest.mark.asyncio
async def test_briefing_falls_back_to_live_when_the_stored_row_is_malformed(
    mocker: MockerFixture, home_dependencies
) -> None:
    mocker.patch(
        "backend.api.features.executions.activity_gate.is_feature_enabled",
        AsyncMock(return_value=True),
    )
    _patch_stored_briefing(mocker, {"unexpected": "shape"})

    dashboard = await build_home_dashboard(user_id="user-1")

    assert dashboard.briefing.source == "live"
    assert dashboard.briefing.outcomes[0].title == "Booked the flight."


@pytest.mark.asyncio
async def test_briefing_is_live_when_no_row_exists_yet(
    mocker: MockerFixture, home_dependencies
) -> None:
    """A user who signed up after 9am has no briefing to anchor on."""
    mocker.patch(
        "backend.api.features.executions.activity_gate.is_feature_enabled",
        AsyncMock(return_value=True),
    )

    dashboard = await build_home_dashboard(user_id="user-1")

    assert dashboard.briefing.source == "live"
    assert [outcome.id for outcome in dashboard.briefing.outcomes] == ["exec-1"]


@pytest.mark.asyncio
async def test_persisted_summaries_are_scrubbed_when_the_activity_flag_is_off(
    mocker: MockerFixture, home_dependencies
) -> None:
    """Summaries are stored regardless of the flag, so the persisted path has
    to scrub them too — otherwise the card leaks what the gate hides."""
    mocker.patch(
        "backend.api.features.executions.activity_gate.is_feature_enabled",
        AsyncMock(return_value=False),
    )
    mocker.patch(
        "backend.api.features.home.service.is_feature_enabled",
        AsyncMock(return_value=False),
    )
    _patch_stored_briefing(mocker, _stored_briefing().model_dump(mode="json"))

    dashboard = await build_home_dashboard(user_id="user-1")

    assert dashboard.briefing.source == "persisted"
    assert dashboard.briefing.outcomes[0].title == "Inbox triage finished"
    assert dashboard.briefing.outcomes[0].summary == "Completed successfully."


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
