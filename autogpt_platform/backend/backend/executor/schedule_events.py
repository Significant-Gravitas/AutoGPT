"""Best-effort recording of schedules: created, and how each fire ended.

The scheduler is the one place every schedule passes through (REST route,
copilot tools, expert installs), so it records each new schedule once: a
``schedule.created`` ActivityEvent for the Home feed and the ``analytics.*``
views, and a ``schedule_created`` product event. Mirrors
``backend.executor.activity_events`` for run completions: recording the
schedule must never fail or delay the schedule.

It also records the outcome of every scheduled copilot follow-up fire
(``schedule.dispatched`` / ``.skipped`` / ``.dropped`` / ``.failed``). APScheduler
removes a one-shot job the moment it fires, so without this row a follow-up
that was dropped at the concurrency cap and one that ran look identical to
``list_schedules``: both are simply gone. The row is what lets the next turn
tell the user "your 06:12 check did not run" instead of "nothing is pending".

The activity event goes through the DatabaseManager RPC, whose client keeps
retrying for a long time while the service is unreachable. That write is
therefore handed to a small background worker pool: the schedule call returns
immediately and a DatabaseManager outage costs a log line, not a stalled
scheduler. The pool is bounded so an outage queues records instead of
spawning a thread per schedule, and its workers are joined at interpreter
exit so an in-flight write is not cut off by shutdown.
"""

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Lock
from typing import Literal, cast, get_args

from pydantic import BaseModel

from backend.data.activity_event import (
    SCHEDULE_FIRE_EVENT_TYPES,
    ActivityEvent,
    ActivityEventDraft,
)
from backend.util import product_analytics
from backend.util.clients import get_database_manager_client
from backend.util.product_analytics import ScheduleTarget

logger = logging.getLogger(__name__)


class ScheduleCreatedRecord(BaseModel):
    user_id: str
    schedule_id: str
    title: str
    target: ScheduleTarget
    expert_id: str | None = None
    organization_id: str | None = None
    cron: str | None = None
    run_at: datetime | None = None
    graph_id: str | None = None
    session_id: str | None = None
    next_run_time: str | None = None


def record_schedule_created(record: ScheduleCreatedRecord) -> None:
    try:
        _submit(lambda: _write_activity_event(record))
    except Exception:
        logger.warning(
            "Failed to submit schedule.created for %s",
            record.schedule_id,
            exc_info=True,
        )
    product_analytics.track_schedule_created(
        user_id=record.user_id,
        schedule_id=record.schedule_id,
        target=record.target,
        expert_id=record.expert_id,
        cron=record.cron,
        run_at=record.run_at,
        graph_id=record.graph_id,
        session_id=record.session_id,
    )


FollowupOutcomeStatus = Literal["dispatched", "skipped", "dropped", "failed"]
# Every status has a matching ``schedule.<status>`` event type and vice versa;
# the data layer owns the list so the Home feed can exclude them without
# importing the scheduler.
assert tuple(f"schedule.{s}" for s in get_args(FollowupOutcomeStatus)) == (
    SCHEDULE_FIRE_EVENT_TYPES
)

# Statuses under which the user did NOT get the turn they were promised.
UNDELIVERED_OUTCOME_STATUSES: tuple[FollowupOutcomeStatus, ...] = (
    "skipped",
    "dropped",
    "failed",
)

_MESSAGE_PREVIEW_CHARS = 120


class ScheduleFiredRecord(BaseModel):
    """What the scheduler knows when a copilot follow-up fire ends."""

    user_id: str
    schedule_id: str | None
    status: FollowupOutcomeStatus
    reason: str
    fired_at: datetime
    message: str
    session_id: str | None = None
    expert_id: str | None = None
    organization_id: str | None = None
    cron: str | None = None
    scheduled_for: datetime | None = None
    # Set when the fire was deferred into a fresh one-shot job (concurrency
    # cap, transient expert lookup) so a reader can follow the chain.
    retry_schedule_id: str | None = None


class ScheduleFireOutcome(BaseModel):
    """A persisted fire outcome, read back from its activity event."""

    event_id: str
    schedule_id: str | None
    status: FollowupOutcomeStatus
    reason: str
    fired_at: datetime
    message_preview: str
    session_id: str | None = None
    expert_id: str | None = None
    cron: str | None = None
    scheduled_for: str | None = None
    retry_schedule_id: str | None = None


def message_preview(message: str) -> str:
    flat = " ".join(message.split())
    if len(flat) <= _MESSAGE_PREVIEW_CHARS:
        return flat
    return flat[: _MESSAGE_PREVIEW_CHARS - 1] + "…"


def fire_outcome_title(record: ScheduleFiredRecord) -> str:
    label = f'Follow-up "{message_preview(record.message)}"'
    if record.status == "dispatched":
        return f"{label} ran"
    return f"{label} {record.status}: {record.reason}"


def record_schedule_fired(record: ScheduleFiredRecord) -> None:
    """Persist a follow-up fire outcome. Never raises."""
    try:
        _submit(lambda: _write_fire_event(record))
    except Exception:
        logger.warning(
            "Failed to submit schedule.%s for %s",
            record.status,
            record.schedule_id,
            exc_info=True,
        )


def _write_fire_event(record: ScheduleFiredRecord) -> None:
    try:
        get_database_manager_client().create_activity_event(
            user_id=record.user_id,
            draft=ActivityEventDraft(
                category="SCHEDULE",
                event_type=f"schedule.{record.status}",
                title=fire_outcome_title(record),
                schedule_id=record.schedule_id,
                expert_id=record.expert_id,
                organization_id=record.organization_id,
                session_id=record.session_id,
                data={
                    "status": record.status,
                    "reason": record.reason,
                    "fired_at": record.fired_at.isoformat(),
                    "scheduled_for": (
                        record.scheduled_for.isoformat()
                        if record.scheduled_for
                        else None
                    ),
                    "cron": record.cron,
                    "is_recurring": record.cron is not None,
                    "message_preview": message_preview(record.message),
                    "retry_schedule_id": record.retry_schedule_id,
                },
            ),
        )
    except Exception:
        logger.warning(
            "Failed to record schedule.%s for %s",
            record.status,
            record.schedule_id,
            exc_info=True,
        )


def fire_outcome_from_event(event: ActivityEvent) -> ScheduleFireOutcome | None:
    """Read a fire outcome back from its row; ``None`` for any other event."""
    if event.event_type not in SCHEDULE_FIRE_EVENT_TYPES:
        return None
    status = cast(FollowupOutcomeStatus, event.event_type.removeprefix("schedule."))
    data = event.data

    def text(key: str) -> str | None:
        value = data.get(key)
        return value if isinstance(value, str) else None

    return ScheduleFireOutcome(
        event_id=event.id,
        schedule_id=event.schedule_id,
        status=status,
        reason=text("reason") or "",
        fired_at=event.created_at,
        message_preview=text("message_preview") or "",
        session_id=event.session_id,
        expert_id=event.expert_id,
        cron=text("cron"),
        scheduled_for=text("scheduled_for"),
        retry_schedule_id=text("retry_schedule_id"),
    )


def _write_activity_event(record: ScheduleCreatedRecord) -> None:
    try:
        get_database_manager_client().create_activity_event(
            user_id=record.user_id,
            draft=ActivityEventDraft(
                category="SCHEDULE",
                event_type="schedule.created",
                title=record.title,
                schedule_id=record.schedule_id,
                expert_id=record.expert_id,
                organization_id=record.organization_id,
                session_id=record.session_id,
                object_id=record.graph_id,
                data={
                    "target": record.target,
                    "cron": record.cron,
                    "run_at": record.run_at.isoformat() if record.run_at else None,
                    "next_run_time": record.next_run_time,
                    "is_recurring": record.cron is not None,
                },
            ),
        )
    except Exception:
        logger.warning(
            "Failed to record schedule.created for %s",
            record.schedule_id,
            exc_info=True,
        )


_executor: ThreadPoolExecutor | None = None
_executor_lock = Lock()


def _submit(work: Callable[[], None]) -> None:
    """Run *work* off the caller's thread. Tests replace this to run inline."""
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="schedule-created-activity-event"
            )
        executor = _executor
    executor.submit(work)
