import logging
from datetime import datetime, timezone
from typing import Optional

from backend.util.clients import get_scheduler_client

logger = logging.getLogger(__name__)
ScheduleScopeKey = tuple[str, int, str | None, str | None]


async def _fetch_schedule_info(
    user_id: str,
    graph_id: Optional[str] = None,
    *,
    exact_scope: bool = False,
) -> dict[str | ScheduleScopeKey, str]:
    """Fetch a map of graph ID to earliest next-run time.

    When ``graph_id`` is provided, narrow the scheduler query to that graph.
    """
    try:
        scheduler_client = get_scheduler_client()
        schedules = await scheduler_client.get_graph_execution_schedules(
            graph_id=graph_id,
            user_id=user_id,
        )
        earliest: dict[str | ScheduleScopeKey, tuple[datetime, str]] = {}
        for schedule in schedules:
            parsed = _parse_iso_datetime(schedule.next_run_time)
            if parsed is None:
                continue
            key: str | ScheduleScopeKey = (
                (
                    schedule.graph_id,
                    schedule.graph_version,
                    schedule.organization_id or None,
                    schedule.team_id,
                )
                if exact_scope
                else schedule.graph_id
            )
            current = earliest.get(key)
            if current is None or parsed < current[0]:
                earliest[key] = (parsed, schedule.next_run_time)
        return {key: iso for key, (_, iso) in earliest.items()}
    except Exception:
        logger.warning("Failed to fetch schedules for library agents", exc_info=True)
        return {}


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
