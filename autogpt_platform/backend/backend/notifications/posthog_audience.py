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

from collections.abc import Iterable
from datetime import datetime

from pydantic import BaseModel

from backend.data.notifications import SubscriberField
from backend.notifications.audience_enrichment import YES, email_type, signin_method
from backend.notifications.subscriber_fields import Fields, utc_moment
from backend.util import posthog_client
from backend.util.posthog_events import PostHogEvent

CHECKOUT_OPENED_AT = "checkout_opened_at"

# Sent as they are now, under MailerLite's own keys.
_ALWAYS_SET = (SubscriberField.EMAIL_TYPE, SubscriberField.SIGNIN_METHOD)
_COUNTRY_FIELDS = (
    SubscriberField.COUNTRY,
    SubscriberField.COUNTRY_CODE,
    SubscriberField.COUNTRY_SOURCE,
)


class PersonUpdate(BaseModel):
    """One `$set` event: values that replace the person's, and values that
    only fill in ones the person doesn't hold yet."""

    to_set: dict[str, str] = {}
    to_set_once: dict[str, str] = {}

    def properties(self) -> dict[str, dict[str, str]]:
        return {
            key: values
            for key, values in (("$set", self.to_set), ("$set_once", self.to_set_once))
            if values
        }


def set_account_properties(user_id: str, email: str, signup_method: str | None) -> None:
    """A new account's email type and sign-in method. Never raises."""
    send(user_id, account_update(email, [signup_method] if signup_method else []))


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
    country = {f.value: value for f in _COUNTRY_FIELDS if (value := fields.get(f))}
    billed = fields.get(SubscriberField.COUNTRY_SOURCE) == "stripe"
    (update.to_set if billed else update.to_set_once).update(country)
    opened = utc_moment(opened_at).isoformat(timespec="seconds")
    (update.to_set if first_open else update.to_set_once)[CHECKOUT_OPENED_AT] = opened
    return update


def send(user_id: str, update: PersonUpdate) -> None:
    """Queue `update` as one `$set` event. Never raises, and sends nothing
    when analytics is off or there is nothing to set."""
    if properties := update.properties():
        posthog_client.capture(user_id, PostHogEvent.SET_PERSON_PROPERTIES, properties)
