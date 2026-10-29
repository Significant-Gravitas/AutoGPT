"""Tests for the schedule.created record written when a schedule is persisted."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from backend.executor import schedule_events
from backend.executor.schedule_events import (
    ScheduleCreatedRecord,
    record_schedule_created,
)
from backend.executor.scheduler import (
    CopilotTurnJobArgs,
    GraphExecutionJobArgs,
    _record_copilot_turn_schedule_created,
    _record_graph_schedule_created,
)


@pytest.fixture
def db_client(monkeypatch: pytest.MonkeyPatch) -> Mock:
    client = Mock()
    monkeypatch.setattr(schedule_events, "get_database_manager_client", lambda: client)
    # The write normally runs on a background thread; run it inline so the
    # assertions below see it.
    monkeypatch.setattr(schedule_events, "_submit", lambda work: work())
    return client


@pytest.fixture
def track_created(monkeypatch: pytest.MonkeyPatch) -> Mock:
    tracker = Mock()
    monkeypatch.setattr(
        schedule_events.product_analytics, "track_schedule_created", tracker
    )
    return tracker


def _job(next_run_time: datetime | None = datetime(2026, 9, 3, 7, tzinfo=UTC)):
    return SimpleNamespace(id="sched-1", name="x", next_run_time=next_run_time)


def _graph_args(expert_id: str | None = None) -> GraphExecutionJobArgs:
    return GraphExecutionJobArgs(
        schedule_id="sched-1",
        user_id="user-1",
        graph_id="graph-1",
        graph_version=3,
        agent_name="Daily digest",
        cron="0 7 * * *",
        input_data={},
        organization_id="org-1",
        expert_id=expert_id,
    )


def test_graph_schedule_records_activity_event_and_product_event(
    db_client: Mock, track_created: Mock
) -> None:
    _record_graph_schedule_created(_graph_args(), _job(), title="Daily digest")

    db_client.create_activity_event.assert_called_once()
    kwargs = db_client.create_activity_event.call_args.kwargs
    draft = kwargs["draft"]
    assert kwargs["user_id"] == "user-1"
    assert draft.category == "SCHEDULE"
    assert draft.event_type == "schedule.created"
    assert draft.title == "Daily digest"
    assert draft.schedule_id == "sched-1"
    assert draft.object_id == "graph-1"
    assert draft.organization_id == "org-1"
    assert draft.session_id is None
    assert draft.data["target"] == "agent"
    assert draft.data["cron"] == "0 7 * * *"
    assert draft.data["is_recurring"] is True
    assert draft.data["next_run_time"] == "2026-09-03T07:00:00+00:00"

    track_created.assert_called_once()
    props = track_created.call_args.kwargs
    assert props["target"] == "agent"
    assert props["graph_id"] == "graph-1"
    assert props["schedule_id"] == "sched-1"
    assert "name" not in props


def test_expert_schedule_targets_expert(db_client: Mock, track_created: Mock) -> None:
    _record_graph_schedule_created(_graph_args(expert_id="expert-1"), _job(), title="t")

    draft = db_client.create_activity_event.call_args.kwargs["draft"]
    assert draft.expert_id == "expert-1"
    assert draft.data["target"] == "expert"
    assert track_created.call_args.kwargs["target"] == "expert"


def test_copilot_turn_schedule_records_session_and_run_at(
    db_client: Mock, track_created: Mock
) -> None:
    run_at = datetime(2026, 9, 2, 15, 30, tzinfo=UTC)
    args = CopilotTurnJobArgs(
        schedule_id="sched-2",
        user_id="user-1",
        session_id="session-1",
        message="Check whether CI is green",
        run_at=run_at,
    )

    _record_copilot_turn_schedule_created(
        args, _job(next_run_time=None), title="Check CI"
    )

    draft = db_client.create_activity_event.call_args.kwargs["draft"]
    assert draft.session_id == "session-1"
    assert draft.object_id is None
    assert draft.organization_id is None
    assert draft.data["target"] == "autopilot"
    assert draft.data["is_recurring"] is False
    assert draft.data["run_at"] == run_at.isoformat()
    assert draft.data["next_run_time"] is None
    assert track_created.call_args.kwargs["target"] == "autopilot"
    assert track_created.call_args.kwargs["session_id"] == "session-1"


def test_db_failure_never_breaks_schedule_creation(
    db_client: Mock, track_created: Mock
) -> None:
    db_client.create_activity_event.side_effect = RuntimeError("db down")

    record_schedule_created(
        ScheduleCreatedRecord(
            user_id="user-1", schedule_id="sched-1", title="t", target="agent"
        )
    )

    track_created.assert_called_once()


def test_missing_schedule_id_is_skipped(db_client: Mock, track_created: Mock) -> None:
    args = _graph_args().model_copy(update={"schedule_id": None})

    _record_graph_schedule_created(args, _job(), title="t")

    db_client.create_activity_event.assert_not_called()
    track_created.assert_not_called()
