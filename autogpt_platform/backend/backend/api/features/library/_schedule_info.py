import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel

from backend.util.clients import get_scheduler_client
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()


class ScheduleInfo(BaseModel):
    """Earliest next-run time per graph ID, and whether the scheduler answered.

    ``available=False`` means the schedules are unknown, not absent: the
    frontend must not read an empty map as "this user has no schedules".
    """

    next_runs: dict[str, str] = {}
    available: bool = True


async def _fetch_schedule_info(
    user_id: str, graph_id: Optional[str] = None
) -> ScheduleInfo:
    """Fetch a map of graph ID to earliest next-run time.

    When ``graph_id`` is provided, narrow the scheduler query to that graph.

    Schedule badges decorate library reads; they must never hold the read
    itself hostage. A slow or overloaded scheduler costs the badges after
    ``schedule_read_timeout_seconds``, not the page after the 300s RPC default.
    """
    try:
        scheduler_client = get_scheduler_client()
        schedules = await asyncio.wait_for(
            scheduler_client.get_graph_execution_schedules(
                graph_id=graph_id,
                user_id=user_id,
            ),
            timeout=settings.config.schedule_read_timeout_seconds,
        )
        earliest: dict[str, tuple[datetime, str]] = {}
        for schedule in schedules:
            parsed = _parse_iso_datetime(schedule.next_run_time)
            if parsed is None:
                continue
            current = earliest.get(schedule.graph_id)
            if current is None or parsed < current[0]:
                earliest[schedule.graph_id] = (parsed, schedule.next_run_time)
        return ScheduleInfo(
            next_runs={graph_id: iso for graph_id, (_, iso) in earliest.items()}
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Scheduler did not answer within "
            f"{settings.config.schedule_read_timeout_seconds}s; library agents "
            "render with schedule info marked unavailable"
        )
        return ScheduleInfo(available=False)
    except Exception:
        logger.warning("Failed to fetch schedules for library agents", exc_info=True)
        return ScheduleInfo(available=False)


def _parse_iso_datetime(value: str) -> Optional[datetime]:
    """Parse an ISO 8601 datetime, tolerating Z and naive UTC forms."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        logger.warning("Failed to parse schedule next_run_time: %s", value)
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed
