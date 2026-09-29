"""The MailerLite trial group holds exactly the customers currently on a trial.

Every transition in and out of a trial already produces exactly one trial
notice, deduped by its own claim, so the group change rides on that notice:
join on start and on un-cancel, leave on cancel, conversion, failed conversion
and end. The backend owns both edges of this group (see `mailerlite.py`).

A conversion is also the customer's first paid subscription, so it joins the
paying audience the way a first checkout does.
"""

import logging
from collections.abc import Mapping

from pydantic import ValidationError

from backend.data.db_accessors import user_db
from backend.data.notifications import AudienceAction, AudienceEventModel
from backend.notifications.queue import queue_audience_change
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

_TRIAL_GROUP_CHANGES: Mapping[str, AudienceAction] = {
    "started": AudienceAction.ADD_TRIAL,
    "resumed": AudienceAction.ADD_TRIAL,
    "canceled": AudienceAction.REMOVE_TRIAL,
    "converted": AudienceAction.REMOVE_TRIAL,
    "payment_failed": AudienceAction.REMOVE_TRIAL,
    "ended": AudienceAction.REMOVE_TRIAL,
}


async def queue_trial_group_change(kind: str, user_id: str, email: str) -> None:
    """Queue the trial group change for this notice, if it has one.

    Raises when it cannot be queued, so the caller releases the notice claim
    and Stripe's retry makes the change. Both changes are idempotent, so a
    retry that repeats one is harmless. Nothing is queued while the trial group
    is not configured: a change could never succeed and would only retry.
    """
    action = _TRIAL_GROUP_CHANGES.get(kind)
    if action is None or not settings.config.mailerlite_trial_group_id:
        return
    event = _event(action, email, user_id)
    if event is None:
        return
    result = await queue_audience_change(event)
    if not result.success:
        raise RuntimeError(f"Could not queue {action.value}: {result.message}")


async def join_paying_audience(user_id: str, email: str) -> None:
    """The onboarding tour for a first subscription, else the changelog.

    The conversion notice stands in for the subscription welcome, so this takes
    the same welcome claim a first checkout does: a later resubscription is
    then treated as the returning customer it is. Called once the notice is
    out, so, like the tour enrolment after a welcome, a failure is reported
    rather than raised: a Stripe retry would find the notice claimed and do
    nothing, so raising could only fail the webhook.
    """
    try:
        first = await user_db().claim_welcome_email(user_id)
        action = AudienceAction.ENROLL_TOUR if first else AudienceAction.ADD_CHANGELOG
        event = _event(action, email, user_id)
        if event is None:
            return
        result = await queue_audience_change(event)
    except Exception:
        logger.exception(f"Trial for user {user_id} converted but was not enrolled")
        return
    if not result.success:
        logger.error(
            f"Trial for user {user_id} converted but {action.value} could not be "
            f"queued: {result.message}"
        )


def _event(
    action: AudienceAction, email: str, user_id: str
) -> AudienceEventModel | None:
    """None for an address MailerLite would refuse, such as a reserved domain:
    no retry can ever deliver that change, so it must not hold up the notice."""
    try:
        return AudienceEventModel(action=action, email=email, user_id=user_id)
    except ValidationError:
        logger.warning(
            f"User {user_id}'s email cannot be a MailerLite subscriber; "
            f"skipping {action.value}"
        )
        return None
