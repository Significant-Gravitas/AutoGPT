"""Put everyone who has opened Stripe checkout where the live checkout event
would have: in the checkout openers group, with their status, dates and the
fields GTM segments on (`audience_enrichment`).

Only accounts with at least one Stripe Checkout Session are openers. Nobody
else is written, so the accounts that never reached checkout stay out of
MailerLite. An opener who opted out of marketing stays out too (`consent.py`):
they are counted, never planned.

Each person is written with one subscriber upsert, one at a time and paced,
never a /batch: MailerLite processes an upsert-only batch as an import, and
imports were seen to answer OK for existing subscribers without storing their
fields. The pace leaves MailerLite's per-account limit for the live consumer.

The live consumer keeps writing while a run lasts its hours, so the plan is
only a list of who to visit: each person is read again just before their
write and merged by the same rules as the live event, so a German IP the
live event recorded mid-run is never overwritten by the plan's older guess.
That also makes a run resumable: an interrupted or repeated one picks up
where the last one stopped.
"""

import asyncio
import logging
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping

from pydantic import BaseModel

from backend.data.notifications import SubscriberField
from backend.notifications.audience_enrichment import checkout_fields, merge_with_held
from backend.notifications.consent import marketing_allowed
from backend.notifications.mailerlite import (
    API_BASE,
    _client,
    _find_subscriber,
    _headers,
    _payload,
)
from backend.notifications.mailerlite_backfill import (
    BatchAnswer,
    Subscription,
    _refusal,
)
from backend.notifications.mailerlite_field_backfill import (
    Current,
    Person,
    _normalise,
    _valid,
    desired,
)
from backend.notifications.subscriber_fields import Fields

logger = logging.getLogger(__name__)

# One person every two seconds, a read and a write each: half MailerLite's
# 120 calls a minute, so the live consumer keeps the rest.
WRITE_INTERVAL_SECONDS = 2.0


class Opener(BaseModel):
    """An account that opened Stripe checkout, with what we know of it."""

    person: Person
    # The first Checkout Session's creation, as a Unix timestamp.
    opened_at: int
    signin_providers: list[str] = []
    # The Stripe customer's billing address country, when it has one.
    stripe_country: str | None = None


class OpenerChange(BaseModel):
    opener: Opener
    # Only the fields that differ from MailerLite's.
    fields: Fields
    # Not a MailerLite subscriber yet: writing creates one.
    new: bool
    # Not in the checkout openers group yet.
    joins: bool


class OpenerPlan(BaseModel):
    changes: list[OpenerChange]
    openers: int
    # Addresses MailerLite would refuse, such as a reserved domain.
    invalid: int
    # Openers who refused marketing, left out of the plan and the tallies.
    opted_out: int
    # What the openers end up with, for the report.
    country_sources: dict[str, int]
    countries: dict[str, int]
    email_types: dict[str, int]
    signin_methods: dict[str, int]
    exclude_de_at: int


def wanted(opener: Opener) -> Fields:
    """Everything the opener should hold: the checkout segmentation, with the
    status and dates worked out from Stripe in place of the live default."""
    status, standing_fields = desired(opener.person)
    fields = checkout_fields(
        email=opener.person.email,
        created_at=opener.person.created_at,
        opened_at=opener.opened_at,
        signin_providers=opener.signin_providers,
        timezone=opener.person.timezone,
        stripe_country=opener.stripe_country,
        status=status.value,
    )
    return {**fields, **standing_fields}


def plan(
    openers: list[Opener], current: Current, members: Mapping[str, str]
) -> OpenerPlan:
    changes: list[OpenerChange] = []
    invalid = opted_out = 0
    sources: Counter[str] = Counter()
    countries: Counter[str] = Counter()
    email_types: Counter[str] = Counter()
    methods: Counter[str] = Counter()
    exclude = 0
    for opener in openers:
        if not marketing_allowed(opener.person):
            opted_out += 1
            continue
        email = opener.person.email
        if not _valid(email):
            invalid += 1
            continue
        key = email.strip().lower()
        held = current.get(key)
        fields = _merged(opener, held)
        final = {**(held or {}), **{k.value: v for k, v in fields.items()}}
        sources[str(final.get(SubscriberField.COUNTRY_SOURCE.value) or "unknown")] += 1
        countries[str(final.get(SubscriberField.COUNTRY_CODE.value) or "unknown")] += 1
        email_types[str(final.get(SubscriberField.EMAIL_TYPE.value))] += 1
        methods[str(final.get(SubscriberField.SIGNIN_METHOD.value) or "unknown")] += 1
        exclude += final.get(SubscriberField.EXCLUDE_DE_AT.value) == "yes"
        differ = _changed(fields, held)
        joins = key not in members
        if differ or joins:
            changes.append(
                OpenerChange(
                    opener=opener, fields=differ, new=held is None, joins=joins
                )
            )
    return OpenerPlan(
        changes=changes,
        openers=len(openers),
        invalid=invalid,
        opted_out=opted_out,
        country_sources=dict(sources),
        countries=dict(countries),
        email_types=dict(email_types),
        signin_methods=dict(methods),
        exclude_de_at=exclude,
    )


def _merged(opener: Opener, held: Mapping[str, object] | None) -> Fields:
    """Everything the opener should hold, by the live event's rules against
    what MailerLite holds; the status is Stripe's, so it always wins."""
    return merge_with_held(wanted(opener), held or {}, keep_held_status=False)


def _changed(fields: Fields, held: Mapping[str, object] | None) -> Fields:
    """The fields MailerLite does not already hold: all of them for someone
    it does not hold at all."""
    return {
        field: value
        for field, value in fields.items()
        if held is None or _normalise(field, held.get(field.value)) != value
    }


async def apply(
    changes: list[OpenerChange],
    group_id: str,
    on_progress: Callable[[int, int], None] | None = None,
    *,
    refresh: Callable[[str], Awaitable[list[Subscription]]] | None = None,
) -> tuple[int, int, int]:
    """Visit each planned opener: read them again from MailerLite and, with
    `refresh`, their subscriptions again from Stripe, and write what they
    still need as one subscriber upsert into the group. The status the plan
    took from Stripe hours earlier is never written over a newer one.

    Returns (succeeded, failed, skipped); a failure is logged with
    MailerLite's reason and left for the next run, and someone who needs
    nothing any more is skipped."""
    succeeded = failed = skipped = 0
    for index, change in enumerate(changes):
        if index:
            await asyncio.sleep(WRITE_INTERVAL_SECONDS)
        if on_progress and index and index % 100 == 0:
            on_progress(index, len(changes))
        email = change.opener.person.email
        opener = change.opener
        try:
            if refresh and opener.person.stripe_customer_id:
                person = opener.person.model_copy(
                    update={
                        "subscriptions": await refresh(opener.person.stripe_customer_id)
                    }
                )
                opener = opener.model_copy(update={"person": person})
            subscriber = await _find_subscriber(email)
        except Exception:
            failed += 1
            logger.warning(
                "Re-reading checkout opener %s failed; the next run retries it",
                _refusal(email, BatchAnswer(code=0)),
            )
            continue
        held = None if subscriber is None else (subscriber.get("fields") or {})
        fields = _changed(_merged(opener, held), held)
        if not fields and not change.joins:
            skipped += 1
            continue
        body: dict = {"email": email, "groups": [group_id]}
        if fields:
            body["fields"] = _payload(fields)
        try:
            response = await _client().post(
                f"{API_BASE}/subscribers", headers=_headers(), json=body
            )
        except Exception:
            # One person's network failure must not end a run of thousands.
            failed += 1
            logger.warning(
                "Checkout opener write failed for %s; the next run retries it",
                _refusal(email, BatchAnswer(code=0)),
            )
            continue
        if response.status in (200, 201, 202, 204):
            succeeded += 1
        else:
            failed += 1
            try:
                answer_body = response.json()
            except Exception:
                answer_body = None
            logger.warning(
                "Checkout opener write failed for %s; the next run retries it",
                _refusal(email, BatchAnswer(code=response.status, body=answer_body)),
            )
    if on_progress:
        on_progress(len(changes), len(changes))
    return succeeded, failed, skipped
