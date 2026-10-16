"""Give accounts from before SECRT-2895 the PostHog person properties the live
code sends since (`posthog_audience`): every account its email type and
sign-in method, and every checkout opener its first open, country and
exclude_de_at.

An opener is an account whose Stripe customer has a Checkout Session, as for
`mailerlite-checkout-backfill`, and is worked out the same way
(`audience_enrichment.checkout_fields`), from the Stripe billing country or
else the browser's timezone. The live code's own rules keep a run from making
PostHog worse: the IP country a live checkout saw was never stored, so a
timezone country is only filled in and never replaces it, and neither does an
exclude_de_at "no". The first open is the exception and is set outright: Stripe
holds every Checkout Session, so its first one is the first open, even where
the live code saw a later one first.

Nobody is left out: PostHog is analytics, not marketing. A repeated run sends
the same values, so it is harmless.
"""

from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from functools import partial

from posthog import Posthog
from pydantic import BaseModel

from backend.data.notifications import SubscriberField
from backend.notifications import posthog_audience
from backend.notifications.audience_enrichment import YES, checkout_fields
from backend.notifications.posthog_audience import CHECKOUT_OPENED_AT, PersonUpdate
from backend.util.posthog_client import send_each

NONE = "none"


class Account(BaseModel):
    user_id: str
    email: str
    created_at: datetime
    timezone: str | None = None
    stripe_customer_id: str | None = None


class AccountUpdate(BaseModel):
    user_id: str
    update: PersonUpdate


class AudiencePlan(BaseModel):
    updates: list[AccountUpdate]
    # Accounts whose Stripe customer has a Checkout Session.
    openers: int
    # What the run sends, for the report. A `$set_once` value only lands where
    # the person holds none yet, so these aren't what PostHog will hold.
    email_types: dict[str, int]
    signin_methods: dict[str, int]
    # Openers only.
    country_sources: dict[str, int]
    exclude_de_at: int


def plan(
    accounts: list[Account],
    providers: Mapping[str, list[str]],
    first_opens: Mapping[str, int],
    billing_countries: Mapping[str, str],
) -> AudiencePlan:
    """`providers` by account id; `first_opens` (Unix timestamps) and
    `billing_countries` by Stripe customer id."""
    updates = [
        AccountUpdate(
            user_id=account.user_id,
            update=_update(
                account,
                providers.get(account.user_id, []),
                first_opens.get(account.stripe_customer_id or ""),
                billing_countries.get(account.stripe_customer_id or ""),
            ),
        )
        for account in accounts
    ]
    return _tally(updates)


def _update(
    account: Account,
    providers: list[str],
    first_open: int | None,
    billing_country: str | None,
) -> PersonUpdate:
    if first_open is None:
        return posthog_audience.account_update(account.email, providers)
    fields = checkout_fields(
        email=account.email,
        created_at=account.created_at,
        opened_at=first_open,
        signin_providers=providers,
        timezone=account.timezone,
        stripe_country=billing_country,
    )
    return posthog_audience.checkout_update(fields, first_open, first_open=True)


def _tally(updates: list[AccountUpdate]) -> AudiencePlan:
    email_types: Counter[str] = Counter()
    signin_methods: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    openers = exclude_de_at = 0
    for change in updates:
        values = {**change.update.to_set_once, **change.update.to_set}
        email_types[values.get(SubscriberField.EMAIL_TYPE.value, NONE)] += 1
        signin_methods[values.get(SubscriberField.SIGNIN_METHOD.value, NONE)] += 1
        if CHECKOUT_OPENED_AT not in values:
            continue
        openers += 1
        sources[values.get(SubscriberField.COUNTRY_SOURCE.value, NONE)] += 1
        if values.get(SubscriberField.EXCLUDE_DE_AT.value) == YES:
            exclude_de_at += 1
    return AudiencePlan(
        updates=updates,
        openers=openers,
        email_types=dict(email_types),
        signin_methods=dict(signin_methods),
        country_sources=dict(sources),
        exclude_de_at=exclude_de_at,
    )


async def send(updates: list[AccountUpdate]) -> int:
    """One `$set` event per account. Returns how many accounts PostHog didn't
    take (`posthog_client.send_each`). The run sends the same values every
    time, so retrying is harmless."""
    return await send_each(
        (change.user_id, partial(_send_update, change)) for change in updates
    )


def _send_update(change: AccountUpdate, client: Posthog) -> bool:
    return posthog_audience.send(change.user_id, change.update, client=client)
