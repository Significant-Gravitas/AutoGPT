"""Where each person stands with us, as MailerLite subscriber fields.

GTM segments on one status and five dates (see `SubscriberField`), plus the
checkout opener's segmentation, which `audience_enrichment` works out. Each
transition below sets the status and the date it is named for, from Stripe's
own timestamp where the event carries one. The dates describe the current or
most recent subscription: a new subscription clears the previous one's
cancellation and end, and a resumed one clears its cancellation. The signup
and trial start dates are never cleared.

A trial that ends without its first payment clearing, whether it was cancelled
or the payment failed, is `trial_canceled`: the person never had a paid
subscription, so `subscription_ended` would count them as churned customers.
`in_trial` is therefore exactly the trial group's membership.
"""

import logging
from datetime import UTC, datetime

from pydantic import ValidationError

from backend.data.notifications import (
    AudienceAction,
    AudienceEventModel,
    SubscriberField,
    SubscriptionStatus,
)
from backend.notifications.queue import queue_audience_change

logger = logging.getLogger(__name__)

Fields = dict[SubscriberField, str | None]


def mailerlite_date(value: datetime | int | float | None) -> str:
    """A Stripe timestamp or a datetime as a MailerLite date, in UTC. Only an
    event without a timestamp of its own falls back to now."""
    if value is None:
        moment = datetime.now(UTC)
    elif isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=UTC)
    else:
        moment = datetime.fromtimestamp(value, UTC)
    return moment.astimezone(UTC).strftime("%Y-%m-%d")


def signed(created_at: datetime) -> Fields:
    return {
        SubscriberField.STATUS: SubscriptionStatus.SIGNED.value,
        SubscriberField.SIGNUP: mailerlite_date(created_at),
    }


def trial_started(trial_start: int | None) -> Fields:
    return {
        SubscriberField.STATUS: SubscriptionStatus.IN_TRIAL.value,
        SubscriberField.TRIAL_STARTED: mailerlite_date(trial_start),
    }


def trial_resumed() -> Fields:
    return {SubscriberField.STATUS: SubscriptionStatus.IN_TRIAL.value}


def trial_canceled() -> Fields:
    return {SubscriberField.STATUS: SubscriptionStatus.TRIAL_CANCELED.value}


def subscribed(started: int | None) -> Fields:
    return {
        SubscriberField.STATUS: SubscriptionStatus.SUBSCRIBED.value,
        SubscriberField.SUBSCRIPTION_STARTED: mailerlite_date(started),
        SubscriberField.SUBSCRIPTION_CANCELED: None,
        SubscriberField.SUBSCRIPTION_ENDED: None,
    }


def subscription_resumed() -> Fields:
    return {
        SubscriberField.STATUS: SubscriptionStatus.SUBSCRIBED.value,
        SubscriberField.SUBSCRIPTION_CANCELED: None,
    }


def subscription_canceled(canceled_at: int | None) -> Fields:
    return {
        SubscriberField.STATUS: SubscriptionStatus.SUBSCRIPTION_CANCELED.value,
        SubscriberField.SUBSCRIPTION_CANCELED: mailerlite_date(canceled_at),
    }


def subscription_ended(ended_at: int | None) -> Fields:
    return {
        SubscriberField.STATUS: SubscriptionStatus.SUBSCRIPTION_ENDED.value,
        SubscriberField.SUBSCRIPTION_ENDED: mailerlite_date(ended_at),
    }


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
