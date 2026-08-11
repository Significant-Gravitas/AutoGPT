"""The Alert engine.

Alerts fire only when the platform is blocked on the human, and the rules that
keep them worth reading live here:

* **Debounce and coalesce.** New conditions are held for ten minutes, then
  everything pending goes out as one email.
* **Never twice in 24 hours.** The same cause does not re-alert within a day;
  if it persists it escalates into the next Briefing's attention block.
* **Two per day, hard cap.** Overflow folds into the Briefing. An alert channel
  that can flood is an alert channel people mute.
* **Cancel solved problems.** A condition that clears during the debounce
  window cancels the send.
"""

import logging
from datetime import datetime, timedelta, timezone

from prisma.enums import NotificationType

from backend.data import alerts as alerts_db
from pydantic import BaseModel

from backend.data.notifications import AlertData, AlertPrimary, NotificationEventModel
from backend.notifications.alert_causes import SEVERITY, BaseCause, parse_cause
from backend.util.logging import TruncatedLogger
from backend.util.settings import Settings

logger = TruncatedLogger(logging.getLogger(__name__), prefix="[Alerts]")
settings = Settings()

ALERT_DEBOUNCE = timedelta(minutes=10)
MAX_ALERT_EMAILS_PER_DAY = 2


async def raise_alert(user_id: str, cause_key: str, cause: BaseCause) -> None:
    """Record that the platform is blocked on this user for `cause_key`.

    Emitters call this instead of sending anything: whether it becomes an
    email, a line in a coalesced email, or an attention card in the next
    Briefing is the engine's decision, not the caller's.
    """
    await alerts_db.raise_condition(
        user_id=user_id,
        cause=cause.cause,
        cause_key=cause_key,
        data=cause.model_dump(mode="json"),
    )


async def resolve_alert(user_id: str, cause_key: str) -> None:
    """The problem went away. During the debounce window this cancels the send
    outright — the user reconnected Gmail from their phone and never needs to
    hear that it was disconnected."""
    if await alerts_db.resolve_condition(user_id, cause_key):
        logger.info("Alert condition %s resolved for user %s", cause_key, user_id)


async def matured_alert_user_ids() -> list[str]:
    """Users whose pending conditions have sat out the debounce window."""
    return await alerts_db.get_users_with_matured_alerts(
        datetime.now(tz=timezone.utc) - ALERT_DEBOUNCE
    )


class BuiltAlert(BaseModel):
    """The email plus the conditions it covers, so the caller can mark them
    sent only once the message is safely on the queue."""

    data: AlertData
    condition_ids: list[str]


async def build_alert_email(user_id: str, alerts_enabled: bool) -> BuiltAlert | None:
    """Assemble one coalesced Alert for a user, or defer everything into the
    Briefing when the cap is hit or alerts are switched off.

    Returns None when nothing should be emailed; in that case the conditions
    have already been marked deferred, so the Briefing will carry them.
    """
    pending = await alerts_db.get_pending_conditions(user_id)
    if not pending:
        return None

    condition_ids = [row.id for row in pending]

    if not alerts_enabled:
        await alerts_db.mark_deferred(condition_ids)
        return None

    sent_today = await alerts_db.count_alerts_sent_since(user_id, _start_of_day())
    if sent_today >= MAX_ALERT_EMAILS_PER_DAY:
        logger.info(
            "User %s hit the %d-alert daily cap; folding %d conditions into the "
            "next briefing",
            user_id,
            MAX_ALERT_EMAILS_PER_DAY,
            len(condition_ids),
        )
        await alerts_db.mark_deferred(condition_ids)
        return None

    causes = sorted(
        (parse_cause(row.cause, row.data) for row in pending),
        key=lambda c: SEVERITY[c.cause],
    )
    primary, rest = causes[0], causes[1:]
    base_url = settings.config.frontend_base_url or settings.config.platform_base_url

    data = AlertData(
        timestamp_label=_timestamp_label(),
        primary=AlertPrimary(
            headline=primary.headline,
            subject=primary.subject_line,
            preheader=primary.body,
            body=primary.body,
            cta_label=primary.cta_label,
            cta_url=f"{base_url}{primary.cta_path}",
            microcopy=primary.microcopy,
            facts=primary.facts,
        ),
        also=[c.also_item(base_url) for c in rest],
        also_label="Also waiting" if rest else None,
    )
    return BuiltAlert(data=data, condition_ids=condition_ids)


async def mark_alert_sent(condition_ids: list[str]) -> None:
    """Called once the email is on the queue. Marking earlier would lose the
    conditions if the publish failed; marking later is the reason the same
    cause can't re-alert for 24 hours."""
    await alerts_db.mark_sent(condition_ids)


def alert_event(user_id: str, data: AlertData) -> NotificationEventModel[AlertData]:
    return NotificationEventModel[AlertData](
        user_id=user_id, type=NotificationType.ALERT, data=data
    )


def _start_of_day() -> datetime:
    now = datetime.now(tz=timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _timestamp_label() -> str:
    """Absolute, never a duration: the email is read hours later, and a stale
    relative time is a wrong time."""
    now = datetime.now(tz=timezone.utc)
    return f"{now.strftime('%a')} {now.day} {now.strftime('%b')}, {now.strftime('%H:%M')}"
