"""Best-effort recording of new schedules, whichever surface created them.

The scheduler is the one place every schedule passes through (REST route,
copilot tools, expert installs), so it records each new schedule once: a
``schedule.created`` ActivityEvent for the Home feed and the ``analytics.*``
views, and a ``schedule_created`` product event. Mirrors
``backend.executor.activity_events`` for run completions: recording the
schedule must never fail or delay the schedule.

The activity event goes through the DatabaseManager RPC, whose client keeps
retrying for a long time while the service is unreachable. That write is
therefore handed to a background thread: the schedule call returns
immediately and a DatabaseManager outage costs a log line, not a stalled
scheduler.
"""

import logging
import threading
from collections.abc import Callable
from datetime import datetime

from pydantic import BaseModel

from backend.data.activity_event import ActivityEventDraft
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
    _submit(lambda: _write_activity_event(record))
    product_analytics.track_schedule_created(
        user_id=record.user_id,
        schedule_id=record.schedule_id,
        target=record.target,
        expert_id=record.expert_id,
        cron=record.cron,
        run_at=record.run_at,
        graph_id=record.graph_id,
        session_id=record.session_id,
        name=record.title,
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


def _submit(work: Callable[[], None]) -> None:
    """Run *work* off the caller's thread. Tests replace this to run inline."""
    threading.Thread(
        target=work, name="schedule-created-activity-event", daemon=True
    ).start()
