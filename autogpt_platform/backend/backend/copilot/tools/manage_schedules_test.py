"""Tests for ListSchedulesTool and DeleteScheduleTool."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.tools.manage_schedules import (
    DeleteScheduleTool,
    ListSchedulesTool,
    PauseScheduleTool,
    ResumeScheduleTool,
    ScheduleDeletedResponse,
    ScheduleListResponse,
    ScheduleToggledResponse,
)
from backend.copilot.tools.models import ErrorResponse
from backend.data.activity_event import ActivityEvent
from backend.executor.scheduler import CopilotTurnJobInfo, GraphExecutionJobInfo

from ._test_data import make_session

_USER = "test-user-schedules"
_SCHEDULES_PATH = "backend.copilot.tools.manage_schedules"


@pytest.fixture(autouse=True)
def outcome_events():
    """The list tool also reads recent follow-up fire outcomes; default to
    none so the pre-existing listing tests see exactly what they always did."""
    store = MagicMock()
    store.list_activity_events_by_type = AsyncMock(return_value=[])
    with patch(f"{_SCHEDULES_PATH}.activity_event_db", return_value=store):
        yield store


def _fire_event(
    *,
    event_id: str = "evt-1",
    status: str = "dropped",
    schedule_id: str = "cop-gone",
    expert_id: str | None = None,
    session_id: str = "session-xyz",
) -> ActivityEvent:
    return ActivityEvent(
        id=event_id,
        user_id=_USER,
        created_at=datetime(2026, 9, 30, 6, 12, 25, tzinfo=timezone.utc),
        category="SCHEDULE",
        event_type=f"schedule.{status}",
        title="x",
        schedule_id=schedule_id,
        session_id=session_id,
        expert_id=expert_id,
        data={
            "status": status,
            "reason": "the account was at its concurrent-turn limit",
            "scheduled_for": "2026-09-30T06:12:00+00:00",
            "message_preview": "check CI on PR #999",
        },
    )


def _make_graph_info(
    *, schedule_id: str = "sched-1", expert_id: str | None = None
) -> GraphExecutionJobInfo:
    return GraphExecutionJobInfo(
        schedule_id=schedule_id,
        user_id=_USER,
        graph_id="graph-1",
        graph_version=1,
        agent_name="My Schedule",
        cron="*/5 * * * *",
        input_data={"input": "data"},
        input_credentials={},
        id=schedule_id,
        name="My Schedule",
        next_run_time="2026-04-13T12:00:00",
        timezone="UTC",
        expert_id=expert_id,
    )


def _make_copilot_info(
    *, schedule_id: str = "cop-1", expert_id: str | None = None
) -> CopilotTurnJobInfo:
    return CopilotTurnJobInfo(
        schedule_id=schedule_id,
        user_id=_USER,
        session_id="session-xyz",
        message="check CI on PR #999",
        run_at=datetime(2026, 5, 22, 19, 0, tzinfo=timezone.utc),
        id=schedule_id,
        name="copilot turn (session session-)",
        next_run_time="2026-05-22T19:00:00+00:00",
        timezone="UTC",
        expert_id=expert_id,
    )


@pytest.fixture
def list_tool():
    return ListSchedulesTool()


@pytest.fixture
def delete_tool():
    return DeleteScheduleTool()


@pytest.fixture
def session():
    return make_session(_USER)


# ── ListSchedulesTool ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_schedules_no_auth(list_tool, session):
    result = await list_tool._execute(user_id=None, session=session)
    assert isinstance(result, ErrorResponse)
    assert result.error == "auth_required"


@pytest.mark.asyncio
async def test_list_schedules_returns_graph_kind(list_tool, session):
    mock_client = AsyncMock()
    mock_client.get_execution_schedules = AsyncMock(return_value=[_make_graph_info()])

    with patch(
        f"{_SCHEDULES_PATH}.get_scheduler_client",
        return_value=mock_client,
    ):
        result = await list_tool._execute(user_id=_USER, session=session)

    assert isinstance(result, ScheduleListResponse)
    assert len(result.schedules) == 1
    summary = result.schedules[0]
    assert summary.schedule_id == "sched-1"
    assert summary.kind == "graph"
    assert summary.cron == "*/5 * * * *"
    assert summary.graph_id == "graph-1"
    assert summary.graph_version == 1
    assert summary.session_id is None
    assert summary.message is None


@pytest.mark.asyncio
async def test_list_schedules_returns_copilot_turn_kind(list_tool, session):
    mock_client = AsyncMock()
    mock_client.get_execution_schedules = AsyncMock(return_value=[_make_copilot_info()])

    with patch(
        f"{_SCHEDULES_PATH}.get_scheduler_client",
        return_value=mock_client,
    ):
        result = await list_tool._execute(user_id=_USER, session=session)

    assert isinstance(result, ScheduleListResponse)
    assert len(result.schedules) == 1
    summary = result.schedules[0]
    assert summary.schedule_id == "cop-1"
    assert summary.kind == "copilot_turn"
    assert summary.session_id == "session-xyz"
    assert summary.message == "check CI on PR #999"
    assert summary.run_at == "2026-05-22T19:00:00+00:00"
    assert summary.graph_id is None
    assert summary.graph_version is None


@pytest.mark.asyncio
async def test_list_schedules_returns_mixed_kinds(list_tool, session):
    mock_client = AsyncMock()
    mock_client.get_execution_schedules = AsyncMock(
        return_value=[_make_graph_info(), _make_copilot_info()]
    )

    with patch(
        f"{_SCHEDULES_PATH}.get_scheduler_client",
        return_value=mock_client,
    ):
        result = await list_tool._execute(user_id=_USER, session=session)

    assert isinstance(result, ScheduleListResponse)
    assert len(result.schedules) == 2
    assert {s.kind for s in result.schedules} == {"graph", "copilot_turn"}


@pytest.mark.asyncio
async def test_list_schedules_empty(list_tool, session):
    mock_client = AsyncMock()
    mock_client.get_execution_schedules = AsyncMock(return_value=[])

    with patch(
        f"{_SCHEDULES_PATH}.get_scheduler_client",
        return_value=mock_client,
    ):
        result = await list_tool._execute(user_id=_USER, session=session)

    assert isinstance(result, ScheduleListResponse)
    assert len(result.schedules) == 0
    assert "No schedules" in result.message


@pytest.mark.asyncio
async def test_list_schedules_reports_a_dropped_followup_after_the_job_is_gone(
    list_tool, session, outcome_events
):
    """SECRT-2787: the one-shot fired and was dropped, so the scheduler has
    nothing pending — but the outcome record still tells the model the
    check never ran instead of letting it conclude nothing was scheduled."""
    outcome_events.list_activity_events_by_type.return_value = [_fire_event()]
    mock_client = AsyncMock()
    mock_client.get_execution_schedules = AsyncMock(return_value=[])

    with patch(f"{_SCHEDULES_PATH}.get_scheduler_client", return_value=mock_client):
        result = await list_tool._execute(user_id=_USER, session=session)

    assert isinstance(result, ScheduleListResponse)
    assert result.schedules == []
    assert len(result.recent_outcomes) == 1
    outcome = result.recent_outcomes[0]
    assert outcome.schedule_id == "cop-gone"
    assert outcome.status == "dropped"
    assert outcome.reason == "the account was at its concurrent-turn limit"
    assert outcome.scheduled_for == "2026-09-30T06:12:00+00:00"
    assert outcome.fired_at == "2026-09-30T06:12:25+00:00"
    assert outcome.message == "check CI on PR #999"
    assert result.message == (
        "No schedules found. 1 follow-up fire(s) in the last 24h, "
        "1 of which did not run (see recent_outcomes)."
    )
    kwargs = outcome_events.list_activity_events_by_type.call_args.kwargs
    assert kwargs["user_id"] == _USER
    assert set(kwargs["event_types"]) == {
        "schedule.dispatched",
        "schedule.skipped",
        "schedule.dropped",
        "schedule.failed",
    }


@pytest.mark.asyncio
async def test_list_schedules_counts_dispatched_fires_as_delivered(
    list_tool, session, outcome_events
):
    outcome_events.list_activity_events_by_type.return_value = [
        _fire_event(event_id="evt-1", status="dispatched"),
    ]
    mock_client = AsyncMock()
    mock_client.get_execution_schedules = AsyncMock(return_value=[_make_copilot_info()])

    with patch(f"{_SCHEDULES_PATH}.get_scheduler_client", return_value=mock_client):
        result = await list_tool._execute(user_id=_USER, session=session)

    assert isinstance(result, ScheduleListResponse)
    assert result.message == "Found 1 schedule(s). 1 follow-up fire(s) in the last 24h."
    assert result.recent_outcomes[0].status == "dispatched"


@pytest.mark.asyncio
async def test_list_schedules_scopes_outcomes_like_schedules(list_tool, outcome_events):
    """An expert sees only its own follow-ups' outcomes; personal AutoPilot
    sees every outcome on the account — the same rule as the pending list."""
    outcome_events.list_activity_events_by_type.return_value = [
        _fire_event(event_id="evt-a", schedule_id="cop-a", expert_id="expert-a"),
        _fire_event(event_id="evt-b", schedule_id="cop-b", expert_id="expert-b"),
        _fire_event(event_id="evt-c", schedule_id="cop-c", expert_id=None),
    ]
    mock_client = AsyncMock()
    mock_client.get_execution_schedules = AsyncMock(return_value=[])

    with patch(f"{_SCHEDULES_PATH}.get_scheduler_client", return_value=mock_client):
        as_expert = await list_tool._execute(
            user_id=_USER, session=make_session(_USER, expert_id="expert-a")
        )
        as_autopilot = await list_tool._execute(
            user_id=_USER, session=make_session(_USER)
        )

    assert isinstance(as_expert, ScheduleListResponse)
    assert [o.schedule_id for o in as_expert.recent_outcomes] == ["cop-a"]
    assert isinstance(as_autopilot, ScheduleListResponse)
    assert {o.schedule_id for o in as_autopilot.recent_outcomes} == {
        "cop-a",
        "cop-b",
        "cop-c",
    }


@pytest.mark.asyncio
async def test_list_schedules_degrades_when_outcome_read_fails(
    list_tool, session, outcome_events
):
    outcome_events.list_activity_events_by_type.side_effect = RuntimeError("db down")
    mock_client = AsyncMock()
    mock_client.get_execution_schedules = AsyncMock(return_value=[_make_copilot_info()])

    with patch(f"{_SCHEDULES_PATH}.get_scheduler_client", return_value=mock_client):
        result = await list_tool._execute(user_id=_USER, session=session)

    assert isinstance(result, ScheduleListResponse)
    assert len(result.schedules) == 1
    assert result.recent_outcomes == []
    assert result.message == "Found 1 schedule(s)."


@pytest.mark.asyncio
async def test_list_schedules_by_library_agent(list_tool, session):
    mock_agent = MagicMock(graph_id="graph-42")
    mock_client = AsyncMock()
    mock_client.get_execution_schedules = AsyncMock(return_value=[])

    with (
        patch(
            f"{_SCHEDULES_PATH}.get_library_agent",
            new_callable=AsyncMock,
            return_value=mock_agent,
        ),
        patch(
            f"{_SCHEDULES_PATH}.get_scheduler_client",
            return_value=mock_client,
        ),
    ):
        result = await list_tool._execute(
            user_id=_USER,
            session=session,
            library_agent_id="lib-agent-1",
        )

    assert isinstance(result, ScheduleListResponse)
    mock_client.get_execution_schedules.assert_called_once_with(
        graph_id="graph-42", user_id=_USER, include_paused=True
    )


@pytest.mark.asyncio
async def test_list_schedules_library_agent_not_found(list_tool, session):
    from backend.util.exceptions import NotFoundError

    with patch(
        f"{_SCHEDULES_PATH}.get_library_agent",
        new_callable=AsyncMock,
        side_effect=NotFoundError("not found"),
    ):
        result = await list_tool._execute(
            user_id=_USER,
            session=session,
            library_agent_id="missing",
        )

    assert isinstance(result, ErrorResponse)
    assert result.error == "library_agent_not_found"


@pytest.mark.parametrize(
    ("session_expert_id", "expected_ids"),
    [
        (None, ["autopilot", "expert-a-job", "expert-b-job"]),
        ("expert-a", ["expert-a-job"]),
        ("expert-b", ["expert-b-job"]),
    ],
)
@pytest.mark.asyncio
async def test_list_schedules_autopilot_sees_all_experts_see_their_own(
    list_tool, session_expert_id, expected_ids
):
    scoped_session = make_session(_USER, expert_id=session_expert_id)
    mock_client = AsyncMock()
    mock_client.get_execution_schedules = AsyncMock(
        return_value=[
            _make_graph_info(schedule_id="autopilot", expert_id=None),
            _make_graph_info(schedule_id="expert-a-job", expert_id="expert-a"),
            _make_copilot_info(schedule_id="expert-b-job", expert_id="expert-b"),
        ]
    )

    with patch(f"{_SCHEDULES_PATH}.get_scheduler_client", return_value=mock_client):
        result = await list_tool._execute(user_id=_USER, session=scoped_session)

    assert isinstance(result, ScheduleListResponse)
    assert [schedule.schedule_id for schedule in result.schedules] == expected_ids
    assert all(
        schedule.expert_id == schedule.schedule_id.removesuffix("-job")
        for schedule in result.schedules
        if schedule.schedule_id != "autopilot"
    )


# ── DeleteScheduleTool ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_schedule_no_auth(delete_tool, session):
    result = await delete_tool._execute(user_id=None, session=session)
    assert isinstance(result, ErrorResponse)
    assert result.error == "auth_required"


@pytest.mark.asyncio
async def test_delete_schedule_missing_id(delete_tool, session):
    result = await delete_tool._execute(user_id=_USER, session=session)
    assert isinstance(result, ErrorResponse)
    assert result.error == "missing_schedule_id"


@pytest.mark.asyncio
async def test_delete_schedule_success(delete_tool, session):
    mock_client = AsyncMock()
    mock_client.get_execution_schedules = AsyncMock(return_value=[_make_graph_info()])
    mock_client.delete_schedule = AsyncMock()

    with patch(
        f"{_SCHEDULES_PATH}.get_scheduler_client",
        return_value=mock_client,
    ):
        result = await delete_tool._execute(
            user_id=_USER,
            session=session,
            schedule_id="sched-1",
        )

    assert isinstance(result, ScheduleDeletedResponse)
    assert result.schedule_id == "sched-1"
    mock_client.delete_schedule.assert_called_once_with(
        schedule_id="sched-1", user_id=_USER
    )


@pytest.mark.asyncio
async def test_delete_schedule_not_found(delete_tool, session):
    from backend.util.exceptions import NotFoundError

    mock_client = AsyncMock()
    mock_client.get_execution_schedules = AsyncMock(return_value=[_make_graph_info()])
    mock_client.delete_schedule = AsyncMock(side_effect=NotFoundError("Job not found"))

    with patch(
        f"{_SCHEDULES_PATH}.get_scheduler_client",
        return_value=mock_client,
    ):
        result = await delete_tool._execute(
            user_id=_USER,
            session=session,
            schedule_id="missing",
        )

    assert isinstance(result, ErrorResponse)
    assert result.error == "schedule_not_found"


@pytest.mark.asyncio
async def test_delete_schedule_not_authorized(delete_tool, session):
    from backend.util.exceptions import NotAuthorizedError

    mock_client = AsyncMock()
    mock_client.get_execution_schedules = AsyncMock(return_value=[_make_graph_info()])
    mock_client.delete_schedule = AsyncMock(
        side_effect=NotAuthorizedError("wrong user")
    )

    with patch(
        f"{_SCHEDULES_PATH}.get_scheduler_client",
        return_value=mock_client,
    ):
        result = await delete_tool._execute(
            user_id=_USER,
            session=session,
            schedule_id="sched-1",
        )

    assert isinstance(result, ErrorResponse)
    assert result.error == "not_authorized"


@pytest.mark.parametrize(
    ("session_expert_id", "target_expert_id"),
    [
        ("expert-a", None),
        ("expert-a", "expert-b"),
        ("expert-b", "expert-a"),
    ],
)
@pytest.mark.asyncio
async def test_delete_schedule_refuses_cross_expert_job(
    delete_tool, session_expert_id, target_expert_id
):
    scoped_session = make_session(_USER, expert_id=session_expert_id)
    mock_client = AsyncMock()
    mock_client.get_execution_schedules = AsyncMock(
        return_value=[
            _make_graph_info(schedule_id="foreign-job", expert_id=target_expert_id)
        ]
    )
    mock_client.delete_schedule = AsyncMock()

    with patch(f"{_SCHEDULES_PATH}.get_scheduler_client", return_value=mock_client):
        result = await delete_tool._execute(
            user_id=_USER,
            session=scoped_session,
            schedule_id="foreign-job",
        )

    assert isinstance(result, ErrorResponse)
    assert result.error == "schedule_not_found"
    mock_client.delete_schedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_schedule_allows_same_expert_job(delete_tool):
    scoped_session = make_session(_USER, expert_id="expert-a")
    mock_client = AsyncMock()
    mock_client.get_execution_schedules = AsyncMock(
        return_value=[_make_graph_info(schedule_id="expert-job", expert_id="expert-a")]
    )
    mock_client.delete_schedule = AsyncMock()

    with patch(f"{_SCHEDULES_PATH}.get_scheduler_client", return_value=mock_client):
        result = await delete_tool._execute(
            user_id=_USER,
            session=scoped_session,
            schedule_id="expert-job",
        )

    assert isinstance(result, ScheduleDeletedResponse)
    mock_client.delete_schedule.assert_awaited_once_with(
        schedule_id="expert-job", user_id=_USER
    )


@pytest.mark.asyncio
async def test_autopilot_can_delete_an_expert_schedule(delete_tool, session):
    mock_client = AsyncMock()
    mock_client.get_execution_schedules = AsyncMock(
        return_value=[_make_graph_info(schedule_id="expert-job", expert_id="expert-a")]
    )
    mock_client.delete_schedule = AsyncMock()

    with patch(f"{_SCHEDULES_PATH}.get_scheduler_client", return_value=mock_client):
        result = await delete_tool._execute(
            user_id=_USER, session=session, schedule_id="expert-job"
        )

    assert isinstance(result, ScheduleDeletedResponse)
    mock_client.delete_schedule.assert_awaited_once()


@pytest.mark.parametrize(
    ("tool", "method", "session_expert_id"),
    [
        (PauseScheduleTool(), "pause_schedule", None),
        (ResumeScheduleTool(), "resume_schedule", None),
        (PauseScheduleTool(), "pause_schedule", "expert-a"),
    ],
)
@pytest.mark.asyncio
async def test_pause_and_resume_within_scope(tool, method, session_expert_id):
    scoped_session = make_session(_USER, expert_id=session_expert_id)
    mock_client = AsyncMock()
    mock_client.get_execution_schedules = AsyncMock(
        return_value=[_make_graph_info(schedule_id="expert-job", expert_id="expert-a")]
    )
    setattr(mock_client, method, AsyncMock(return_value=True))

    with patch(f"{_SCHEDULES_PATH}.get_scheduler_client", return_value=mock_client):
        result = await tool._execute(
            user_id=_USER, session=scoped_session, schedule_id="expert-job"
        )

    assert isinstance(result, ScheduleToggledResponse)
    getattr(mock_client, method).assert_awaited_once_with("expert-job", _USER)


@pytest.mark.asyncio
async def test_expert_cannot_pause_another_experts_schedule():
    scoped_session = make_session(_USER, expert_id="expert-b")
    mock_client = AsyncMock()
    mock_client.get_execution_schedules = AsyncMock(
        return_value=[_make_graph_info(schedule_id="expert-job", expert_id="expert-a")]
    )
    mock_client.pause_schedule = AsyncMock()

    with patch(f"{_SCHEDULES_PATH}.get_scheduler_client", return_value=mock_client):
        result = await PauseScheduleTool()._execute(
            user_id=_USER, session=scoped_session, schedule_id="expert-job"
        )

    assert isinstance(result, ErrorResponse) and result.error == "schedule_not_found"
    mock_client.pause_schedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_schedules_marks_paused_entries(list_tool, session):
    paused = _make_graph_info(schedule_id="paused-job")
    paused.next_run_time = ""
    mock_client = AsyncMock()
    mock_client.get_execution_schedules = AsyncMock(return_value=[paused])

    with patch(f"{_SCHEDULES_PATH}.get_scheduler_client", return_value=mock_client):
        result = await list_tool._execute(user_id=_USER, session=session)

    assert isinstance(result, ScheduleListResponse)
    assert result.schedules[0].paused is True
    assert (
        mock_client.get_execution_schedules.call_args.kwargs["include_paused"] is True
    )


@pytest.mark.parametrize(
    ("tool", "method"),
    [
        (DeleteScheduleTool(), "delete_schedule"),
        (ResumeScheduleTool(), "resume_schedule"),
    ],
)
@pytest.mark.asyncio
async def test_an_archived_experts_paused_schedule_is_out_of_reach(
    tool, method, session
):
    """detach_expert_triggers pauses rather than deletes so re-hire can restore
    the cadence. Deleting one loses it permanently; resuming one produces a
    schedule that fires and is refused at run time. The REST listing already
    hides these rows, so the tools have to agree."""
    paused = _make_graph_info(schedule_id="archived-job", expert_id="gone")
    paused = paused.model_copy(update={"next_run_time": ""})
    mock_client = AsyncMock()
    mock_client.get_execution_schedules = AsyncMock(return_value=[paused])

    with (
        patch(f"{_SCHEDULES_PATH}.get_scheduler_client", return_value=mock_client),
        patch(
            "backend.api.features.experts.experts_db.active_expert_ids",
            AsyncMock(return_value=set()),
        ),
    ):
        result = await tool._execute(
            user_id=_USER, session=session, schedule_id="archived-job"
        )

    assert isinstance(result, ErrorResponse) and result.error == "schedule_not_found"
    getattr(mock_client, method).assert_not_awaited()


@pytest.mark.asyncio
async def test_a_live_experts_paused_schedule_is_still_reachable(session):
    """The guard keys on the expert being gone, not on the schedule being
    paused — pausing one and resuming it must keep working."""
    paused = _make_graph_info(schedule_id="live-job", expert_id="expert-a")
    paused = paused.model_copy(update={"next_run_time": ""})
    mock_client = AsyncMock()
    mock_client.get_execution_schedules = AsyncMock(return_value=[paused])
    mock_client.resume_schedule = AsyncMock(return_value=True)

    with (
        patch(f"{_SCHEDULES_PATH}.get_scheduler_client", return_value=mock_client),
        patch(
            "backend.api.features.experts.experts_db.active_expert_ids",
            AsyncMock(return_value={"expert-a"}),
        ),
    ):
        result = await ResumeScheduleTool()._execute(
            user_id=_USER, session=session, schedule_id="live-job"
        )

    assert isinstance(result, ScheduleToggledResponse)
    mock_client.resume_schedule.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_hides_an_archived_experts_paused_schedule(list_tool, session):
    """Listing one the mutation tools refuse is worse than not listing it: the
    model would hand its id to resume_schedule and be told it does not exist."""
    live = _make_graph_info(schedule_id="live", expert_id="expert-a")
    archived = _make_graph_info(schedule_id="archived", expert_id="gone").model_copy(
        update={"next_run_time": ""}
    )
    mock_client = AsyncMock()
    mock_client.get_execution_schedules = AsyncMock(return_value=[live, archived])

    with (
        patch(f"{_SCHEDULES_PATH}.get_scheduler_client", return_value=mock_client),
        patch(
            "backend.api.features.experts.experts_db.active_expert_ids",
            AsyncMock(return_value={"expert-a"}),
        ),
    ):
        result = await list_tool._execute(user_id=_USER, session=session)

    assert isinstance(result, ScheduleListResponse)
    assert [s.schedule_id for s in result.schedules] == ["live"]


@pytest.mark.asyncio
async def test_list_keeps_a_live_experts_paused_schedule(list_tool, session):
    """The guard keys on the expert being gone, not on the schedule being
    paused — a paused row still needs to be listable to be resumed."""
    paused = _make_graph_info(schedule_id="paused", expert_id="expert-a").model_copy(
        update={"next_run_time": ""}
    )
    mock_client = AsyncMock()
    mock_client.get_execution_schedules = AsyncMock(return_value=[paused])

    with (
        patch(f"{_SCHEDULES_PATH}.get_scheduler_client", return_value=mock_client),
        patch(
            "backend.api.features.experts.experts_db.active_expert_ids",
            AsyncMock(return_value={"expert-a"}),
        ),
    ):
        result = await list_tool._execute(user_id=_USER, session=session)

    assert isinstance(result, ScheduleListResponse)
    assert [s.schedule_id for s in result.schedules] == ["paused"]
    assert result.schedules[0].paused is True


@pytest.mark.parametrize(
    ("tool", "method", "changed", "expect_event"),
    [
        (PauseScheduleTool(), "pause_schedule", True, True),
        (PauseScheduleTool(), "pause_schedule", False, False),
        (ResumeScheduleTool(), "resume_schedule", True, True),
        (ResumeScheduleTool(), "resume_schedule", False, False),
    ],
)
@pytest.mark.asyncio
async def test_a_no_op_toggle_leaves_no_activity_event(
    tool, method, changed, expect_event, session
):
    """The scheduler returns False when the schedule was already in that state.
    The activity log is append-only, so pausing an already-paused schedule must
    not leave a schedule.paused entry the user never caused."""
    mock_client = AsyncMock()
    mock_client.get_execution_schedules = AsyncMock(
        return_value=[_make_graph_info(schedule_id="sched-1")]
    )
    setattr(mock_client, method, AsyncMock(return_value=changed))

    with patch(f"{_SCHEDULES_PATH}.get_scheduler_client", return_value=mock_client):
        result = await tool._execute(
            user_id=_USER, session=session, schedule_id="sched-1"
        )

    assert isinstance(result, ScheduleToggledResponse)
    assert result.changed is changed
    assert (tool.activity_event(session, result) is not None) is expect_event
