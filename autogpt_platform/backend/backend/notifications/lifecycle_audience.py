"""The subscription status and dates in MailerLite, from the lifecycle
snapshot PostHog gets too (SECRT-2795).

One mapping (`data/posthog_lifecycle.py`) decides the status for both, so
both agree on who is a paying subscriber. Two ways in:

- live: each lifecycle sync (`data/posthog_lifecycle_sync.py`), which runs
  after every write that can change the status, queues the fields here;
- the field backfill (`mailerlite_field_backfill.py`), which also catches up
  anyone a live write missed. The daily lifecycle sweep does not write
  MailerLite: one read and one write per account would use up MailerLite's
  rate limit every day for changes the live path already made.

MailerLite holds checkout openers, so only an account with a Stripe customer
is queued, and the consumer writes only someone MailerLite already has
(`mailerlite.record_lifecycle`). Nothing is queued for an account that opted
out of marketing or that a signal places in Iran or Russia (`consent.py`).
"""

import logging
from datetime import datetime

from pydantic import BaseModel

from backend.data.db_accessors import posthog_lifecycle_db
from backend.data.notifications import AudienceAction, SubscriberField
from backend.data.posthog_lifecycle import LifecycleSnapshot
from backend.notifications.consent import audience_change_allowed
from backend.notifications.subscriber_fields import (
    Fields,
    mailerlite_date,
    queue_fields,
)
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()


class LifecycleContact(BaseModel):
    """The account a lifecycle sync is for, as consent reads it."""

    id: str
    email: str
    timezone: str | None = None
    marketing_opt_out_at: datetime | None = None
    # The Iranian or Russian country a checkout recorded (`consent.py`).
    excluded_country: str | None = None
    stripe_customer_id: str | None = None


def lifecycle_fields(snapshot: LifecycleSnapshot) -> Fields:
    """The snapshot as MailerLite fields: the status, and each date as a day.
    A date that doesn't apply is cleared, so none from an earlier state
    lingers."""
    return {
        SubscriberField.STATUS: snapshot.subscription_status,
        SubscriberField.SIGNUP: _day(snapshot.signup_at),
        SubscriberField.TRIAL_STARTED: _day(snapshot.trial_started_at),
        SubscriberField.SUBSCRIPTION_STARTED: _day(snapshot.subscription_started_at),
        SubscriberField.SUBSCRIPTION_CANCELED: _day(snapshot.subscription_canceled_at),
        SubscriberField.SUBSCRIPTION_ENDED: _day(snapshot.subscription_ended_at),
    }


def mailerlite_lifecycle_enabled() -> bool:
    """MailerLite's half of a sync can have someone to write: it holds
    checkout openers, and only a stack with Stripe has any. Whether MailerLite
    itself is on is not asked here: only the notification service holds its
    settings, and it drops the change when MailerLite is off."""
    return bool(settings.secrets.stripe_api_key)


async def queue_lifecycle_fields(
    contact: LifecycleContact, snapshot: LifecycleSnapshot
) -> None:
    """Queue the snapshot's fields for MailerLite. Reports rather than
    raises: it rides on a sync that must never fail its caller."""
    if not contact.stripe_customer_id:
        return
    if not audience_change_allowed(
        contact, AudienceAction.LIFECYCLE_FIELDS, (contact.excluded_country,)
    ):
        return
    await queue_fields(
        contact.id,
        contact.email,
        lifecycle_fields(snapshot),
        AudienceAction.LIFECYCLE_FIELDS,
    )


async def resync_after_group_change(user_id: str) -> None:
    """Ask for one more lifecycle sync after a group change that can create
    the subscriber (joining the tour, the changelog or the trial group). The
    sync for the same transition can reach MailerLite first and find nobody
    to write, as when the checkout openers group is off; this one queues its
    fields behind the group change, on the same one-at-a-time queue. Never
    raises: the group change is already queued."""
    try:
        await posthog_lifecycle_db().request_posthog_lifecycle_sync(user_id)
    except Exception:
        logger.warning(
            f"Could not request a lifecycle sync for user {user_id}", exc_info=True
        )


def _day(moment: datetime | None) -> str | None:
    return mailerlite_date(moment) if moment else None
