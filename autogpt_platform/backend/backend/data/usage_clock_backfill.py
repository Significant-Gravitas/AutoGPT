"""Copy every live Stripe subscription's billing period into the usage clock.

Run once after deploying the billing-clock limits and before switching them
on, and by the periodic tier sweep on every run after that::

    poetry run usage-clock-backfill [--dry-run]

It pages through every active and trialing subscription, as the sweep does,
and saves the period for the user who owns each one. It never writes a tier,
skips admin-managed (ENTERPRISE) accounts and trial enrolments, and is safe
to re-run: the clock only moves forwards.
"""

import argparse
import asyncio
import logging

import stripe
from pydantic import BaseModel

from backend.data import db
from backend.data.db import query_raw_with_schema
from backend.data.stripe_client import stripe_call, stripe_list_items
from backend.data.usage_clock import billing_period, save_billing_clock

logger = logging.getLogger(__name__)

_PAGE_SIZE = 100


class ClockBackfillSummary(BaseModel):
    subscriptions: int = 0
    saved: int = 0
    # Already had this period or a later one.
    unchanged: int = 0
    # No user for the customer, an ENTERPRISE account, a trial enrolment, a
    # subscription that names another user, or one with no usable period.
    skipped: int = 0
    failed: int = 0


async def backfill_billing_clocks(*, dry_run: bool = False) -> ClockBackfillSummary:
    summary = ClockBackfillSummary()
    users = await _users_by_customer()
    for status in ("active", "trialing"):
        page = await stripe_call(
            stripe.Subscription.list_async, status=status, limit=_PAGE_SIZE
        )
        async for subscription in stripe_list_items(page):
            await _copy_one(dict(subscription), users, summary, dry_run)
    logger.info(f"Billing clock backfill (dry_run={dry_run}): {summary}")
    return summary


async def _copy_one(
    subscription: dict,
    users: dict[str, str],
    summary: ClockBackfillSummary,
    dry_run: bool,
) -> None:
    summary.subscriptions += 1
    metadata = subscription.get("metadata") or {}
    user_id = users.get(str(subscription.get("customer") or ""))
    if (
        user_id is None
        or metadata.get("trial_enrollment_id")
        or metadata.get("user_id", user_id) != user_id
        or billing_period(subscription) is None
    ):
        summary.skipped += 1
        return
    if dry_run:
        summary.saved += 1
        return
    try:
        saved = await save_billing_clock(user_id, subscription)
    except Exception:
        summary.failed += 1
        logger.exception(f"Could not save the billing clock of user {user_id[:8]}")
        return
    if saved:
        summary.saved += 1
    else:
        summary.unchanged += 1


async def _users_by_customer() -> dict[str, str]:
    """Stripe customer id to user id, for every account Stripe bills."""
    rows = await query_raw_with_schema(
        """SELECT "id", "stripeCustomerId" AS customer
        FROM {schema_prefix}"User"
        WHERE "stripeCustomerId" IS NOT NULL
            AND "subscriptionTier"::text <> 'ENTERPRISE'"""
    )
    return {row["customer"]: row["id"] for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy every live Stripe subscription's billing period "
        "into the usage clock."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count what would be saved without writing anything.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    summary = asyncio.run(_run(args.dry_run))
    print(summary.model_dump_json(indent=2))
    raise SystemExit(1 if summary.failed else 0)


async def _run(dry_run: bool) -> ClockBackfillSummary:
    await db.connect()
    try:
        return await backfill_billing_clocks(dry_run=dry_run)
    finally:
        await db.disconnect()
