from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from autogpt_libs.auth.models import RequestContext
from pytest_mock import MockerFixture

from backend.copilot.briefing.models import BriefingContent, BriefingRunItem
from backend.copilot.scheduled_chats.service import PausedScheduledChat
from backend.data.execution import ExecutionStatus, GraphExecutionMeta
from backend.data.execution_cost_summary import UserExecutionCostSummary

from .service import _get_pending_questions, build_home_dashboard


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
    # Keep owner-scoped Home sources on the same fake database boundary.
    mocker.patch(
        "backend.api.features.home.service.chat_db.get_sessions_with_pending_question",
        AsyncMock(return_value=[]),
    )
    mocker.patch(
        "backend.api.features.home.service.chat_db.get_session_titles",
        AsyncMock(return_value={}),
    )
    mocker.patch(
        "backend.api.features.home.service.activity_db.list_activity_events",
        AsyncMock(return_value=[]),
    )
    mocker.patch(
        "backend.api.features.home.service.scheduled_chats.list_paused_scheduled_chats",
        AsyncMock(return_value=[]),
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
async def test_briefing_rejects_a_row_with_negative_totals(
    mocker: MockerFixture, home_dependencies
) -> None:
    """The counts drive what the card claims happened, and a stored row is
    otherwise taken as canonical — a negative total is a corrupt row, so it
    has to fail validation and drop home onto the live path."""
    corrupt = _stored_briefing().model_dump(mode="json") | {"completed_total": -3}
    _patch_stored_briefing(mocker, corrupt)

    dashboard = await build_home_dashboard(user_id="user-1")

    assert dashboard.briefing.source == "live"
    assert dashboard.briefing.outcomes[0].title == "Booked the flight."


@pytest.mark.asyncio
async def test_briefing_anchors_on_a_row_the_job_could_not_deliver(
    mocker: MockerFixture, home_dependencies
) -> None:
    """`delivered_at=None` means the content was stored but the thread post
    failed. The job redelivers that same stored content, so it is still the
    canonical story — going live here would drift from the pending message."""
    mocker.patch(
        "backend.api.features.home.service.briefing_db.get_briefing_for_date",
        AsyncMock(
            return_value=MagicMock(
                id="briefing-1",
                delivered_at=None,
                content=_stored_briefing().model_dump(mode="json"),
            )
        ),
    )

    dashboard = await build_home_dashboard(user_id="user-1")

    assert dashboard.briefing.source == "persisted"
    assert dashboard.briefing.outcomes[0].title == "Filed the report."


@pytest.mark.asyncio
async def test_briefing_falls_back_to_live_when_the_stored_row_is_malformed(
    mocker: MockerFixture, home_dependencies
) -> None:
    _patch_stored_briefing(mocker, {"unexpected": "shape"})

    dashboard = await build_home_dashboard(user_id="user-1")

    assert dashboard.briefing.source == "live"
    assert dashboard.briefing.outcomes[0].title == "Booked the flight."


@pytest.mark.asyncio
async def test_briefing_falls_back_to_live_when_the_lookup_raises(
    mocker: MockerFixture, home_dependencies
) -> None:
    """A briefing the page cannot read is not worth a 500 — /home is the
    landing page, and every other tile on it is still fine."""
    mocker.patch(
        "backend.api.features.home.service.briefing_db.get_briefing_for_date",
        AsyncMock(side_effect=Exception("database is down")),
    )

    dashboard = await build_home_dashboard(user_id="user-1")

    assert dashboard.briefing.source == "live"
    assert dashboard.briefing.outcomes[0].title == "Booked the flight."


@pytest.mark.asyncio
async def test_briefing_is_live_when_no_row_exists_yet(
    mocker: MockerFixture, home_dependencies
) -> None:
    """A user who signed up after 9am has no briefing to anchor on."""

    dashboard = await build_home_dashboard(user_id="user-1")

    assert dashboard.briefing.source == "live"
    assert [outcome.id for outcome in dashboard.briefing.outcomes] == ["exec-1"]


@pytest.mark.asyncio
async def test_briefing_date_comes_from_the_requests_own_clock(
    mocker: MockerFixture, home_dependencies
) -> None:
    """Loading the source data can cross local midnight. The row looked up has
    to be the one for the date the rest of the dashboard was composed against —
    a second clock read would fetch tomorrow's row for today's dashboard."""
    just_before_midnight = datetime(2026, 8, 10, 23, 59, 59, tzinfo=timezone.utc)
    clock = MagicMock(wraps=datetime)
    # Any read after the first lands on the next day; only one is allowed.
    clock.now.side_effect = [
        just_before_midnight,
        just_before_midnight + timedelta(seconds=2),
    ]
    mocker.patch("backend.api.features.home.service.datetime", clock)
    lookup = AsyncMock(return_value=None)
    mocker.patch(
        "backend.api.features.home.service.briefing_db.get_briefing_for_date", lookup
    )

    dashboard = await build_home_dashboard(user_id="user-1")

    assert lookup.await_args.args[1] == date(2026, 8, 10)
    assert dashboard.generated_at == just_before_midnight


@pytest.mark.asyncio
async def test_activity_summary_surfaces_when_flag_enabled(
    mocker: MockerFixture, home_dependencies
) -> None:

    dashboard = await build_home_dashboard(user_id="user-1")

    assert dashboard.briefing.outcomes[0].title == "Booked the flight."


@pytest.mark.asyncio
async def test_schedules_stay_owner_scoped_inside_an_organization(
    mocker: MockerFixture, home_dependencies
) -> None:
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

    await build_home_dashboard(user_id="user-1", ctx=_org_ctx(is_org_owner=True))

    # Executions, reviews and cost totals are personal, so a teammate's schedule
    # would show an upcoming run whose outcome never lands anywhere else.
    scheduler.get_execution_schedules.assert_awaited_once_with(user_id="user-1")
    get_credit_model.assert_awaited_once_with("user-1", "org-1")


@pytest.mark.asyncio
async def test_scheduler_and_credit_failures_degrade_instead_of_failing_the_page(
    mocker: MockerFixture, home_dependencies
) -> None:
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


@pytest.mark.asyncio
async def test_a_paused_scheduled_chat_needs_attention(
    mocker: MockerFixture, home_dependencies
) -> None:
    mocker.patch(
        "backend.api.features.home.service.scheduled_chats.list_paused_scheduled_chats",
        AsyncMock(
            return_value=[
                PausedScheduledChat(
                    schedule_key="sched-1",
                    title="Discord monitor",
                    session_id="session-1",
                    reason="connection",
                    provider="discord",
                    summary="Paused: can't use Discord",
                    paused_at=datetime.now(timezone.utc),
                )
            ]
        ),
    )

    dashboard = await build_home_dashboard(user_id="user-1")

    assert [item.id for item in dashboard.attention] == ["paused-chat-sched-1"]


@pytest.mark.asyncio
async def test_a_paused_chat_read_that_fails_costs_only_those_items(
    mocker: MockerFixture, home_dependencies
) -> None:
    mocker.patch(
        "backend.api.features.home.service.scheduled_chats.list_paused_scheduled_chats",
        AsyncMock(side_effect=RuntimeError("db")),
    )

    dashboard = await build_home_dashboard(user_id="user-1")

    assert dashboard.attention == []


class TestPendingQuestions:
    """Home reads the user's pending expert questions without a rollout gate."""

    @pytest.mark.asyncio
    async def test_reads_pending_questions_from_chat_db(
        self, mocker: MockerFixture
    ) -> None:
        get_sessions = mocker.patch(
            "backend.api.features.home.service.chat_db.get_sessions_with_pending_question",
            AsyncMock(return_value=["session-info"]),
        )

        result = await _get_pending_questions(user_id="user-1")

        assert result == ["session-info"]
        get_sessions.assert_awaited_once_with("user-1")


def _org_ctx(
    *,
    org_id: str | None = "org-1",
    is_org_owner: bool = False,
    is_org_admin: bool = False,
    is_org_billing_manager: bool = False,
) -> RequestContext:
    return RequestContext(
        user_id="user-1",
        org_id=org_id,
        team_id=None,
        is_org_owner=is_org_owner,
        is_org_admin=is_org_admin,
        is_org_billing_manager=is_org_billing_manager,
        is_team_admin=False,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )


class TestPooledOrgBalanceGate:
    """An org wallet is pooled across its members and MANAGE_BILLING gates it
    on /credits, which 403s a plain member. Home reads the same wallet, so it
    owes that member no figure — and no error either: the rest of the
    dashboard has to arrive intact."""

    @pytest.mark.parametrize(
        ("ctx", "expected_balance"),
        [
            (_org_ctx(is_org_owner=True), 100),
            (_org_ctx(is_org_billing_manager=True), 100),
            (_org_ctx(), None),
            (_org_ctx(is_org_admin=True), None),
        ],
        ids=["owner", "billing-manager", "member", "admin"],
    )
    @pytest.mark.asyncio
    async def test_only_manage_billing_reads_the_pooled_balance(
        self,
        mocker: MockerFixture,
        home_dependencies,
        ctx: RequestContext,
        expected_balance: int | None,
    ) -> None:
        credit_model = MagicMock()
        credit_model.get_credits = AsyncMock(return_value=100)
        get_credit_model = mocker.patch(
            "backend.api.features.home.service.get_credit_model",
            AsyncMock(return_value=credit_model),
        )

        dashboard = await build_home_dashboard(user_id="user-1", ctx=ctx)

        assert dashboard.week.credits_balance == expected_balance
        # Withholding the figure means never reading the wallet, not reading it
        # and dropping the answer.
        assert get_credit_model.await_count == (0 if expected_balance is None else 1)
        # Everything else on the page is owner-scoped and must survive the gate.
        assert dashboard.briefing.outcomes[0].title == "Booked the flight."

    @pytest.mark.asyncio
    async def test_a_personal_org_keeps_its_own_wallet(
        self, mocker: MockerFixture, home_dependencies
    ) -> None:
        """A personal org's one membership row is always isOwner, so the gate
        never fires there and the owner still sees their user wallet."""
        credit_model = MagicMock()
        credit_model.get_credits = AsyncMock(return_value=100)
        get_credit_model = mocker.patch(
            "backend.api.features.home.service.get_credit_model",
            AsyncMock(return_value=credit_model),
        )

        dashboard = await build_home_dashboard(
            user_id="user-1",
            ctx=_org_ctx(org_id="personal-org", is_org_owner=True),
        )

        assert dashboard.week.credits_balance == 100
        get_credit_model.assert_awaited_once_with("user-1", "personal-org")
