"""Where each person stands with us, as MailerLite subscriber fields, and
the queue that carries them.

GTM segments on one status and five dates (see `SubscriberField`), plus the
checkout opener's segmentation, which `audience_enrichment` works out. The
status and dates come from the lifecycle snapshot PostHog gets too
(`lifecycle_audience`), so only a paying subscriber is `subscribed` in either.
"""

import logging
from datetime import UTC, datetime

from pydantic import ValidationError

from backend.data.notifications import (
    AudienceAction,
    AudienceEventModel,
    SubscriberField,
)
from backend.notifications.queue import queue_audience_change

logger = logging.getLogger(__name__)

Fields = dict[SubscriberField, str | None]


def mailerlite_date(value: datetime | float | None) -> str:
    """A Stripe timestamp or a datetime as a MailerLite date, in UTC."""
    return utc_moment(value).strftime("%Y-%m-%d")


def utc_moment(value: datetime | float | None) -> datetime:
    """A Stripe timestamp or a datetime, in UTC. Only an event without a
    timestamp of its own falls back to now."""
    if value is None:
        moment = datetime.now(UTC)
    elif isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=UTC)
    else:
        moment = datetime.fromtimestamp(value, UTC)
    return moment.astimezone(UTC)


def audience_event(
    action: AudienceAction, email: str, user_id: str, fields: Fields | None = None
) -> AudienceEventModel | None:
    """None for an address MailerLite would refuse, such as a reserved domain:
    no retry can ever deliver that change, so it must not hold anything up.

    Whether MailerLite is configured is not asked here: the API server that
    queues these never holds its settings. The notification service decides
    (see `NotificationManager._process_audience_change`)."""
    try:
        return AudienceEventModel(
            action=action,
            email=email,
            user_id=user_id,
            fields=fields or {},
        )
    except ValidationError:
        logger.warning(
            f"User {user_id}'s email cannot be a MailerLite subscriber; "
            f"skipping {action.value}"
        )
        return None


async def queue_fields(
    user_id: str,
    email: str,
    fields: Fields,
    action: AudienceAction = AudienceAction.UPDATE_FIELDS,
) -> None:
    """Queue a field update with no group change. Reports rather than raises:
    it follows a billing email that is already out, and a status must never
    cost one."""
    event = audience_event(action, email, user_id, fields)
    if event is None:
        return
    try:
        result = await queue_audience_change(event)
    except Exception:
        logger.exception(f"Could not queue MailerLite fields for user {user_id}")
        return
    if not result.success:
        logger.error(
            f"Could not queue MailerLite fields for user {user_id}: {result.message}"
        )
