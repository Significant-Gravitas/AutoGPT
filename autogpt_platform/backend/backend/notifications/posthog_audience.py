"""The checkout opener's segmentation on the PostHog person, as in MailerLite.

MailerLite holds what `audience_enrichment` works out for a checkout opener,
and GTM segments on it. PostHog gets the same values as person properties
(SECRT-2895), so funnels, retention and experiments split the same way:

- `email_type` and `signin_method`: every account, from signup;
- `checkout_opened_at`, `country`, `country_code`, `country_source` and
  `exclude_de_at`: every checkout opener. The open is an ISO-8601 UTC moment,
  like the lifecycle dates, where MailerLite holds the day.

The status and its dates are the lifecycle sync's
(`data/posthog_lifecycle_sync.py`), and the profile step sets the role, so
neither is sent from here.

PostHog is analytics, not marketing: nobody is left out for refusing marketing
or because a signal places them in Iran or Russia (`consent.py`).

MailerLite's merge (`audience_enrichment.merge_with_held`) reads the subscriber
before each write. PostHog can't be read on the way, so `$set_once`, which only
fills in what the person doesn't hold yet, keeps the same rules:

- the first checkout open stays;
- exclude_de_at "yes" is set and "no" only filled in, so it never goes back to
  "no";
- a Stripe billing country is set, so it always wins, and an IP or timezone
  country is only filled in, so it never replaces one. The one difference:
  PostHog keeps the first IP or timezone country it gets, where MailerLite
  moves on to a later IP country.
"""

import logging
from collections.abc import Iterable
from datetime import datetime

from posthog import Posthog
from pydantic import BaseModel, Field

from backend.data.notifications import SubscriberField
from backend.notifications.audience_enrichment import (
    COUNTRY_FIELDS,
    YES,
    email_type,
    signin_method,
)
from backend.notifications.subscriber_fields import Fields, utc_moment
from backend.util import posthog_client
from backend.util.posthog_events import PostHogEvent

logger = logging.getLogger(__name__)

CHECKOUT_OPENED_AT = "checkout_opened_at"

# Always `$set`: the account's current values, under MailerLite's own keys.
_ALWAYS_SET = (SubscriberField.EMAIL_TYPE, SubscriberField.SIGNIN_METHOD)

# Every MailerLite field is one or the other, which `posthog_audience_test`
# holds to the whole of `SubscriberField`, so a field added there can't skip
# PostHog unnoticed. The checkout open goes as CHECKOUT_OPENED_AT.
MIRRORED = frozenset(
    {
        *_ALWAYS_SET,
        SubscriberField.EXCLUDE_DE_AT,
        *COUNTRY_FIELDS,
        SubscriberField.CHECKOUT_OPENED,
    }
)
# The lifecycle sync sends the status and its dates, and the profile step the
# role.
NOT_MIRRORED = frozenset(
    {
        SubscriberField.STATUS,
        SubscriberField.SIGNUP,
        SubscriberField.TRIAL_STARTED,
        SubscriberField.SUBSCRIPTION_STARTED,
        SubscriberField.SUBSCRIPTION_CANCELED,
        SubscriberField.SUBSCRIPTION_ENDED,
        SubscriberField.ROLE,
        SubscriberField.ROLE_OTHER,
    }
)


class PersonUpdate(BaseModel):
    """One `$set` event: values that replace the person's, and values that
    only fill in ones the person doesn't hold yet."""

    to_set: dict[str, str] = Field(default_factory=dict)
    to_set_once: dict[str, str] = Field(default_factory=dict)

    def properties(self) -> dict[str, dict[str, str]]:
        return {
            key: values
            for key, values in (("$set", self.to_set), ("$set_once", self.to_set_once))
            if values
        }


def set_account_properties(user_id: str, email: str, signup_method: str | None) -> None:
    """A new account's email type and sign-in method. Never raises: analytics
    must not fail a signup."""
    try:
        providers = [signup_method] if signup_method else []
        send(user_id, account_update(email, providers))
    except Exception:
        logger.warning(f"Could not set the PostHog account properties of {user_id}")


def set_checkout_properties(
    user_id: str, fields: Fields, opened_at: datetime | int | float | None
) -> None:
    """A checkout opener's properties (`checkout_update`). Never raises:
    analytics must not cost the opener their MailerLite change."""
    try:
        send(user_id, checkout_update(fields, opened_at))
    except Exception:
        logger.warning(f"Could not set the PostHog checkout properties of {user_id}")


def account_update(email: str, signin_providers: Iterable[str]) -> PersonUpdate:
    update = PersonUpdate(to_set={SubscriberField.EMAIL_TYPE.value: email_type(email)})
    if method := signin_method(signin_providers):
        update.to_set[SubscriberField.SIGNIN_METHOD.value] = method
    return update


def checkout_update(
    fields: Fields,
    opened_at: datetime | int | float | None,
    *,
    first_open: bool = False,
) -> PersonUpdate:
    """The person properties for a checkout opener's MailerLite `fields`
    (`audience_enrichment.checkout_fields`), opened at `opened_at`, or now when
    that is None. `first_open` is for the backfill: Stripe holds every
    checkout, so the open it names replaces the one PostHog holds."""
    update = PersonUpdate()
    for field in _ALWAYS_SET:
        if value := fields.get(field):
            update.to_set[field.value] = value
    if exclude := fields.get(SubscriberField.EXCLUDE_DE_AT):
        target = update.to_set if exclude == YES else update.to_set_once
        target[SubscriberField.EXCLUDE_DE_AT.value] = exclude
    country = {f.value: value for f in COUNTRY_FIELDS if (value := fields.get(f))}
    billed = fields.get(SubscriberField.COUNTRY_SOURCE) == "stripe"
    (update.to_set if billed else update.to_set_once).update(country)
    opened = utc_moment(opened_at).isoformat(timespec="seconds")
    (update.to_set if first_open else update.to_set_once)[CHECKOUT_OPENED_AT] = opened
    return update


def send(user_id: str, update: PersonUpdate, *, client: Posthog | None = None) -> bool:
    """Queue `update` as one `$set` event, on `client` or else the shared
    client. Never raises. False when the client didn't queue it, analytics
    being off included; an update with nothing to set is not sent and counts
    as done."""
    if not (properties := update.properties()):
        return True
    return posthog_client.capture(
        user_id, PostHogEvent.SET_PERSON_PROPERTIES, properties, client=client
    )
