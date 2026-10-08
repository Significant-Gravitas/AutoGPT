"""The MailerLite trial group holds exactly the customers currently on a trial.

Every transition in and out of a trial already produces exactly one trial
notice, deduped by its own claim, so the group change rides on that notice:
join on start and on un-cancel, leave on cancel, conversion, failed conversion
and end. The backend owns both edges of this group (see `mailerlite.py`).

A conversion is also the customer's first paid subscription, so it joins the
paying audience the way a first checkout does.

The subscriber's status and dates (`subscriber_fields.py`) ride on the same
event. The notification service writes only the fields while the trial group
is not configured.

Nothing is queued for a customer who opted out of marketing (`consent.py`);
their trial notices and claims are unaffected.
"""

import logging
from collections.abc import Callable, Mapping

from backend.data.db_accessors import user_db
from backend.data.notifications import AudienceAction
from backend.notifications import subscriber_fields
from backend.notifications.consent import MarketingContact, audience_change_allowed
from backend.notifications.queue import queue_audience_change
from backend.notifications.subscriber_fields import Fields, audience_event

logger = logging.getLogger(__name__)

_TRIAL_GROUP_CHANGES: Mapping[str, AudienceAction] = {
    "started": AudienceAction.ADD_TRIAL,
    "resumed": AudienceAction.ADD_TRIAL,
    "canceled": AudienceAction.REMOVE_TRIAL,
    "converted": AudienceAction.REMOVE_TRIAL,
    "payment_failed": AudienceAction.REMOVE_TRIAL,
    "ended": AudienceAction.REMOVE_TRIAL,
}

# Each from the live Stripe subscription at the moment the notice applies.
_TRIAL_FIELDS: Mapping[str, Callable[[dict], Fields]] = {
    "started": lambda sub: subscriber_fields.trial_started(sub.get("trial_start")),
    "resumed": lambda _: subscriber_fields.trial_resumed(),
    "canceled": lambda _: subscriber_fields.trial_canceled(),
    "payment_failed": lambda _: subscriber_fields.trial_canceled(),
    "ended": lambda _: subscriber_fields.trial_canceled(),
    # The trial's end is when the first paid period began.
    "converted": lambda sub: subscriber_fields.subscribed(sub.get("trial_end")),
}


async def queue_trial_audience_change(
    kind: str, user: MarketingContact, subscription: dict
) -> None:
    """Queue the trial group change and field update for this notice, if it
    has either.

    Raises when it cannot be queued, so the caller releases the notice claim
    and Stripe's retry makes the change. Both are idempotent, so a retry that
    repeats one is harmless. It is queued whatever this process's MailerLite
    settings: only the notification service holds them.
    """
    action = _TRIAL_GROUP_CHANGES.get(kind)
    if action is None or not audience_change_allowed(user, action):
        return
    fields = _TRIAL_FIELDS[kind](subscription)
    event = audience_event(action, user.email, user.id, fields)
    if event is None:
        return
    result = await queue_audience_change(event)
    if not result.success:
        raise RuntimeError(f"Could not queue {action.value}: {result.message}")


async def join_paying_audience(user: MarketingContact) -> None:
    """The onboarding tour for a first subscription, else the changelog.

    The conversion notice stands in for the subscription welcome, so this takes
    the same welcome claim a first checkout does: a later resubscription is
    then treated as the returning customer it is. A customer who opted out of
    marketing takes the claim too, since it decides who gets the welcome
    email, and only the audience change is skipped. Called once the notice is
    out, so, like the tour enrolment after a welcome, a failure is reported
    rather than raised: a Stripe retry would find the notice claimed and do
    nothing, so raising could only fail the webhook.
    """
    try:
        first = await user_db().claim_welcome_email(user.id)
        action = AudienceAction.ENROLL_TOUR if first else AudienceAction.ADD_CHANGELOG
        if not audience_change_allowed(user, action):
            return
        event = audience_event(action, user.email, user.id)
        if event is None:
            return
        result = await queue_audience_change(event)
    except Exception:
        logger.exception(f"Trial for user {user.id} converted but was not enrolled")
        return
    if not result.success:
        logger.error(
            f"Trial for user {user.id} converted but {action.value} could not be "
            f"queued: {result.message}"
        )
