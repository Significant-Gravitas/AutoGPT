"""The two scheduled passes.

`flush_matured_alerts` runs every minute and empties the debounce window;
`send_due_briefings` runs hourly and catches each user's local ~07:30.

Both deliberately queue rather than send: the queue consumer owns preference
checks, rendering and retries, so one user's failure can't take out the pass.
"""

import logging
from datetime import datetime, timezone
from zoneinfo import available_timezones

from prisma.enums import BriefingFrequency
from prisma.models import User

from backend.notifications import alerts, briefing
from backend.notifications.briefing_period import (
    BRIEFING_HOUR,
    is_briefing_due,
    resolve_zone,
)
from backend.notifications.queue import queue_notification_async
from backend.util.logging import TruncatedLogger

logger = TruncatedLogger(logging.getLogger(__name__), prefix="[NotificationRunner]")

# One page of candidate users per pass, so a very large user base degrades into
# more passes rather than one unbounded query.
CANDIDATE_PAGE_SIZE = 5000


async def flush_matured_alerts() -> None:
    """Ten minutes after a condition is raised, everything pending for that
    user goes out as one email."""
    user_ids = await alerts.matured_alert_user_ids()
    if not user_ids:
        return
    logger.info(f"Flushing matured alerts for {len(user_ids)} users")

    for user_id in user_ids:
        try:
            await _flush_user_alerts(user_id)
        except Exception:
            logger.exception(f"Could not flush alerts for user {user_id}")


async def send_due_briefings() -> None:
    """Assemble briefings for every user whose local briefing hour this is."""
    now = datetime.now(tz=timezone.utc)
    candidates = await _briefing_candidates(now)
    logger.info(f"Considering {len(candidates)} users for a briefing")

    for user in candidates:
        try:
            await _send_user_briefing(user, now)
        except Exception:
            logger.exception(f"Could not build a briefing for user {user.id}")


async def _flush_user_alerts(user_id: str) -> None:
    user = await User.prisma().find_unique(where={"id": user_id})
    if user is None:
        return
    built = await alerts.build_alert_email(user_id, user.alertsEnabled)
    if built is None:
        # Deferred into the next briefing, or nothing left to say.
        return
    result = await queue_notification_async(alerts.alert_event(user_id, built.data))
    if result.success:
        await alerts.mark_alert_sent(built.condition_ids)


async def _send_user_briefing(user: User, now: datetime) -> None:
    frequency = BriefingFrequency(user.briefingFrequency)
    if not is_briefing_due(frequency, user.timezone, now, user.lastBriefingAt):
        return

    built = await briefing.build_briefing(user.id, frequency, user.timezone, now)
    if built is None:
        # Never sent empty: a period with nothing to say produces no email.
        logger.debug(f"Nothing to brief for user {user.id}")
        return

    result = await queue_notification_async(
        briefing.briefing_event(user.id, built.data)
    )
    if not result.success:
        return

    # Only the conditions this briefing actually reported are marked, so one
    # raised while it was being built still gets its turn next period.
    await briefing.mark_attention_reported(built.attention_condition_ids)
    await User.prisma().update(where={"id": user.id}, data={"lastBriefingAt": now})


async def _briefing_candidates(now: datetime) -> list[User]:
    """Users for whom it is currently the briefing hour, locally.

    Filtering on timezone in SQL keeps each hourly pass to roughly a
    twenty-fourth of the user base; `is_briefing_due` then applies the weekly
    and monthly cadence rules.
    """
    return await User.prisma().find_many(
        where={
            "briefingFrequency": {"not": BriefingFrequency.OFF},
            "timezone": {"in": _briefing_hour_timezones(now)},
        },
        take=CANDIDATE_PAGE_SIZE,
        order={"id": "asc"},
    )


def _briefing_hour_timezones(now: datetime) -> list[str]:
    """Every timezone whose local clock currently reads the briefing hour.

    Users who never set one are treated as UTC, matching `resolve_zone`, so
    they are included when it is the briefing hour in UTC.
    """
    zones = [
        name
        for name in available_timezones()
        if now.astimezone(resolve_zone(name)).hour == BRIEFING_HOUR
    ]
    if now.astimezone(timezone.utc).hour == BRIEFING_HOUR:
        zones.append("not-set")
    return zones
