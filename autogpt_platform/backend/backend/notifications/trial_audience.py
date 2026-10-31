"""The MailerLite trial group holds exactly the customers currently on a trial.

Every transition in and out of a trial already produces exactly one trial
notice, deduped by its own claim, so the group change rides on that notice:
join on start and on un-cancel, leave on cancel, conversion, failed conversion
and end. A trial ended by buying another plan sends no notice, so it leaves
the group on its own. The backend owns both edges of this group (see
`mailerlite.py`).

A conversion is also the customer's first paid subscription, so it joins the
paying audience the way a first checkout does.

The subscriber's status and dates are not written here: they come from the
lifecycle sync every trial transition triggers (`lifecycle_audience.py`), by
the same rules as PostHog's. Joining a group can create the subscriber after
that sync found nobody to write, so a join asks for one more sync.

Nothing is queued for a customer who opted out of marketing, or whom a signal
places in Iran or Russia (`consent.py`); their trial notices and claims are
unaffected.
"""

import logging
from collections.abc import Mapping

from backend.data.db_accessors import user_db
from backend.data.notifications import AudienceAction
from backend.notifications.consent import MarketingContact, audience_change_allowed
from backend.notifications.lifecycle_audience import resync_after_group_change
from backend.notifications.queue import queue_audience_change
from backend.notifications.subscriber_fields import audience_event

logger = logging.getLogger(__name__)

_TRIAL_GROUP_CHANGES: Mapping[str, AudienceAction] = {
    "started": AudienceAction.ADD_TRIAL,
    "resumed": AudienceAction.ADD_TRIAL,
    "canceled": AudienceAction.REMOVE_TRIAL,
    "converted": AudienceAction.REMOVE_TRIAL,
    "payment_failed": AudienceAction.REMOVE_TRIAL,
    "ended": AudienceAction.REMOVE_TRIAL,
}


async def queue_trial_audience_change(kind: str, user: MarketingContact) -> None:
    """Queue the trial group change for this notice, if it has one.

    Raises when it cannot be queued, so the caller releases the notice claim
    and Stripe's retry makes the change. It is idempotent, so a retry that
    repeats it is harmless. It is queued whatever this process's MailerLite
    settings: only the notification service holds them.
    """
    action = _TRIAL_GROUP_CHANGES.get(kind)
    if action is None:
        return
    await _queue_group_change(action, user)


async def leave_trial_group(user: MarketingContact) -> None:
    """The trial ended because another plan was bought: leave the trial group.

    No notice or claim rides on it: a repeated removal changes nothing. Raises
    when it cannot be queued, so Stripe retries the event."""
    await _queue_group_change(AudienceAction.REMOVE_TRIAL, user)


async def _queue_group_change(action: AudienceAction, user: MarketingContact) -> None:
    if not audience_change_allowed(user, action):
        return
    event = audience_event(action, user.email, user.id)
    if event is None:
        return
    result = await queue_audience_change(event)
    if not result.success:
        raise RuntimeError(f"Could not queue {action.value}: {result.message}")
    if action == AudienceAction.ADD_TRIAL:
        # A join can create the subscriber; a removal never does.
        await resync_after_group_change(user.id)


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
        return
    await resync_after_group_change(user.id)
