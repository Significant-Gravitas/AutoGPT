"""Queue the checkout-opened audience change for an account.

Called by the checkout routes once a Stripe Checkout Session exists, with the
visitor's IP country, and by the checkout.session.completed webhook with the
Stripe billing country. Firing on every checkout, not once, is safe: the
notification service keeps the first open date, an existing status, and the
strongest country it has seen (`audience_enrichment.merge_with_held`).

It runs in the background and never raises. A checkout must not wait on, or
fail because of, MailerLite bookkeeping; the backfill catches anyone missed.

Nothing is queued for an account that opted out of marketing, or that any
signal places in Iran or Russia, the IP and billing countries included: it
never enters MailerLite (`notifications/consent.py`). An Iranian or Russian
IP or billing country is also recorded on the account, since the trial and
billing events that follow a checkout don't carry it.
"""

import asyncio
import logging
from datetime import datetime

import prisma.models

from backend.data.db import query_raw_with_schema
from backend.data.notifications import AudienceAction
from backend.data.onboarding_role import OnboardingRole, get_onboarding_role
from backend.data.user import get_user_by_id, record_excluded_country
from backend.notifications.audience_enrichment import (
    billing_country,
    checkout_fields,
    excluded_country,
)
from backend.notifications.consent import audience_change_allowed
from backend.notifications.subscriber_fields import queue_fields
from backend.util.retry import func_retry

logger = logging.getLogger(__name__)

# Strong references, so a scheduled task is not garbage-collected mid-flight.
_tasks: set[asyncio.Task] = set()


def schedule_checkout_opened(
    user_id: str,
    *,
    ip_country: str | None = None,
    stripe_country: str | None = None,
    opened_at: datetime | int | None = None,
) -> None:
    """Queue the change in the background, so the checkout response never
    waits on the database or the broker."""
    try:
        task = asyncio.create_task(
            queue_checkout_opened(
                user_id,
                ip_country=ip_country,
                stripe_country=stripe_country,
                opened_at=opened_at,
            )
        )
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)
    except Exception:
        logger.warning(f"Could not schedule the MailerLite checkout for {user_id}")


async def queue_checkout_opened(
    user_id: str,
    *,
    ip_country: str | None = None,
    stripe_country: str | None = None,
    opened_at: datetime | int | None = None,
) -> None:
    try:
        await _record_excluded_country(user_id, (ip_country, stripe_country))
        user = await get_user_by_id(user_id)
        if not audience_change_allowed(
            user, AudienceAction.CHECKOUT_OPENED, (ip_country, stripe_country)
        ):
            return
        fields = checkout_fields(
            email=user.email,
            created_at=user.created_at,
            opened_at=opened_at,
            signin_providers=await signin_providers(user_id),
            timezone=user.timezone,
            stripe_country=stripe_country,
            ip_country=ip_country,
            role=await _role(user_id),
        )
        await queue_fields(user_id, user.email, fields, AudienceAction.CHECKOUT_OPENED)
    except Exception:
        logger.exception(f"Could not queue the MailerLite checkout for {user_id}")


async def record_checkout_completed(session: dict) -> None:
    """A finished checkout carries the billing address, the strongest country
    signal, which also catches a German or Austrian buyer whose IP and
    timezone did not say so. An Iranian or Russian one is recorded before
    this returns, so the webhook runs it before queueing the trial notice,
    whose MailerLite change can only be stopped by that record.

    So it raises when such a country could not be recorded: the webhook fails
    and Stripe retries it, rather than queue the trial notice anyway. Any
    other failure is logged: the webhook must not fail on MailerLite
    bookkeeping."""
    country = billing_country(session)
    excluded = excluded_country((country,))
    try:
        customer = session.get("customer")
        customer_id = customer if isinstance(customer, str) else None
        if not customer_id:
            return
        user = await prisma.models.User.prisma().find_first(
            where={"stripeCustomerId": customer_id}
        )
        if user is None:
            # An organization's customer, or one whose account is gone.
            return
        if excluded:
            await record_excluded_country(user.id, excluded)
        schedule_checkout_opened(
            user.id, stripe_country=country, opened_at=session.get("created")
        )
    except Exception:
        if excluded:
            raise
        logger.exception("Could not record the MailerLite checkout completion")


async def signin_providers(user_id: str) -> list[str]:
    """The account's Better Auth providers: 'google', 'credential' and so on.
    The account id is the auth user id."""
    rows = await query_raw_with_schema(
        'SELECT "providerId" AS provider FROM {schema_prefix}"UserAuthAccount" '
        'WHERE "userId" = $1',
        user_id,
    )
    return [str(row["provider"]) for row in rows if row.get("provider")]


@func_retry
async def _record_excluded_country(
    user_id: str, countries: tuple[str | None, ...]
) -> None:
    """Retried in place: this runs in the background, so a failure has no
    caller to retry it, and the trial and billing events that follow the
    checkout are only stopped by this record. A failure that outlasts the
    retries is logged as an error by `queue_checkout_opened`."""
    if country := excluded_country(countries):
        await record_excluded_country(user_id, country)


async def _role(user_id: str) -> OnboardingRole | None:
    """The role picked in onboarding, if there is one yet. A failed read
    costs the opener their role, never their checkout event."""
    try:
        return await get_onboarding_role(user_id)
    except Exception:
        logger.warning(f"Could not read the onboarding role of {user_id}")
        return None
