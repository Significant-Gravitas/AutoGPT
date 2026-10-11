"""Queue the checkout-opened audience change for an account.

Called by the checkout routes once a Stripe Checkout Session exists, with the
visitor's IP country, and by the checkout.session.completed webhook with the
Stripe billing country. Firing on every checkout, not once, is safe: the
notification service keeps the first open date, an existing status, and the
strongest country it has seen (`audience_enrichment.merge_with_held`).

It runs in the background and never raises. A checkout must not wait on, or
fail because of, MailerLite bookkeeping; the backfill catches anyone missed.

Nothing is queued for an account that opted out of marketing: it never enters
MailerLite (`notifications/consent.py`).
"""

import asyncio
import logging
from datetime import datetime

import prisma.models

from backend.data.db import query_raw_with_schema
from backend.data.notifications import AudienceAction
from backend.data.user import get_user_by_id
from backend.notifications.audience_enrichment import checkout_fields
from backend.notifications.consent import audience_change_allowed
from backend.notifications.subscriber_fields import queue_fields

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
        user = await get_user_by_id(user_id)
        if not audience_change_allowed(user, AudienceAction.CHECKOUT_OPENED):
            return
        fields = checkout_fields(
            email=user.email,
            created_at=user.created_at,
            opened_at=opened_at,
            signin_providers=await signin_providers(user_id),
            timezone=user.timezone,
            stripe_country=stripe_country,
            ip_country=ip_country,
        )
        await queue_fields(user_id, user.email, fields, AudienceAction.CHECKOUT_OPENED)
    except Exception:
        logger.exception(f"Could not queue the MailerLite checkout for {user_id}")


async def record_checkout_completed(session: dict) -> None:
    """A finished checkout carries the billing address, the strongest country
    signal, which also catches a German or Austrian buyer whose IP and
    timezone did not say so. Never raises: the webhook must not fail on it."""
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
        address = (session.get("customer_details") or {}).get("address") or {}
        schedule_checkout_opened(
            user.id,
            stripe_country=address.get("country"),
            opened_at=session.get("created"),
        )
    except Exception:
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
