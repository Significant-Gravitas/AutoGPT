"""List accounts hurt by a failed payment, read-only (SECRT-2873).

Two lists, printed as CSV:

``blocked``  subscriptions that are ``past_due`` or ``unpaid`` with a bill the
             customer can still pay, and the app tier of their account. A
             ``NO_TIER`` account here is locked out of paid features until it
             pays.
``late``     paid renewal or first invoices (``subscription_cycle`` or
             ``subscription_create``) whose failed payment was recorded as
             pausing the plan, paid after a failed attempt, or paid more
             than ``LATE_PAYMENT_AFTER`` after the period they opened had
             begun. Before the billing-period restart shipped, such a
             customer kept the old renewal date, so they paid for the days
             without access. Stripe does not say whether access was cut, so
             this is a list of candidates to check: a bank debit that took
             days but never failed kept its plan throughout.
             ``credited`` says whether the restart's credit note exists,
             ``restarted`` whether the renewal date was also moved; a
             credited row that was not restarted needs a person.

Nothing is written anywhere: Stripe is only listed and retrieved, the
database only read. A restricted Stripe key with read access to customers,
subscriptions, invoices and credit notes is enough, and is what this should
be run with.

Needs DATABASE_URL and STRIPE_API_KEY pointing at the same environment.

Usage::

    poetry run python scripts/list_billing_restart_accounts.py blocked
    poetry run python scripts/list_billing_restart_accounts.py late --since 2026-09-01
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
from datetime import UTC, datetime, timedelta

_BLOCKED_COLUMNS = [
    "user_id",
    "app_tier",
    "customer_id",
    "subscription_id",
    "subscription_status",
    "invoice_id",
    "invoice_status",
    "amount_due",
    "currency",
    "period_start",
]
_LATE_COLUMNS = [
    "user_id",
    "app_tier",
    "customer_id",
    "subscription_id",
    "invoice_id",
    "billing_reason",
    "period_start",
    "paid_at",
    "days_late",
    "amount_paid",
    "currency",
    "credited",
    "restarted",
]


async def main(report: str, since: datetime) -> int:
    if not os.environ.get("DATABASE_URL"):
        raise SystemExit("DATABASE_URL must be set")

    # Imported lazily so --help works without the backend env being loaded.
    import stripe

    from backend.data.db import connect, disconnect
    from backend.util.settings import Settings

    secrets = Settings().secrets
    if not secrets.stripe_api_key:
        raise SystemExit("STRIPE_API_KEY must be set")
    stripe.api_key = secrets.stripe_api_key
    live = secrets.stripe_api_key.startswith(("sk_live", "rk_live"))
    print(f"# Stripe {'live' if live else 'test'} mode, read-only", file=sys.stderr)

    await connect()
    try:
        rows = await (_blocked() if report == "blocked" else _late(since))
        tiers = await _app_tiers({row["customer_id"] for row in rows})
    finally:
        await disconnect()
    columns = _BLOCKED_COLUMNS if report == "blocked" else _LATE_COLUMNS
    writer = csv.DictWriter(sys.stdout, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        user_id, tier = tiers.get(row["customer_id"], ("", ""))
        writer.writerow({**row, "user_id": user_id, "app_tier": tier})
    print(f"# {len(rows)} {report} rows", file=sys.stderr)
    return 0


async def _blocked() -> list[dict]:
    import stripe

    from backend.data.stripe_client import stripe_call, stripe_list_items
    from backend.data.subscription_billing_restart_invoice import current_period_start

    rows = []
    for status in ("past_due", "unpaid"):
        page = await stripe_call(
            stripe.Subscription.list_async,
            status=status,
            limit=100,
            expand=["data.latest_invoice"],
        )
        async for subscription in stripe_list_items(page):
            invoice = subscription.get("latest_invoice") or {}
            if invoice.get("status") not in ("open", "uncollectible"):
                continue
            rows.append(
                {
                    "customer_id": subscription["customer"],
                    "subscription_id": subscription["id"],
                    "subscription_status": status,
                    "invoice_id": invoice.get("id"),
                    "invoice_status": invoice.get("status"),
                    "amount_due": invoice.get("amount_due"),
                    "currency": invoice.get("currency"),
                    "period_start": _iso(current_period_start(dict(subscription))),
                }
            )
    return rows


# Stripe makes the first automatic attempt about an hour after the period
# starts, so an on-time payment lands just after an hour.
LATE_PAYMENT_AFTER = timedelta(hours=3)


def _paid_late(invoice: dict, paused: set[str]) -> tuple[int, int] | None:
    """The period start and payment time of a period-opening invoice that
    was paid late. ``paused`` holds the invoices whose failed payment was
    recorded as pausing the plan; before that record existed, Stripe alone
    decides. A manual payment soon after a failed attempt leaves
    ``attempt_count`` at 1, which only the record catches."""
    from backend.data.subscription_billing_restart import PERIOD_OPENING_REASONS
    from backend.data.subscription_billing_restart_invoice import line_period_start

    if invoice.get("billing_reason") not in PERIOD_OPENING_REASONS:
        return None
    paid_at = (invoice.get("status_transitions") or {}).get("paid_at")
    start = line_period_start(invoice)
    if paid_at is None or start is None:
        return None
    if invoice.get("id") in paused:
        return start, paid_at
    if paid_at - start > LATE_PAYMENT_AFTER.total_seconds():
        return start, paid_at
    return (start, paid_at) if (invoice.get("attempt_count") or 0) > 1 else None


async def _late(since: datetime) -> list[dict]:
    import stripe

    from backend.data.credit import _invoice_subscription_id
    from backend.data.stripe_client import stripe_call, stripe_list_items
    from backend.data.subscription_billing_restart import RESTARTED_FROM_INVOICE_KEY

    rows = []
    paused = await _paused_invoices(since)
    page = await stripe_call(
        stripe.Invoice.list_async,
        status="paid",
        created={"gte": int(since.timestamp())},
        limit=100,
    )
    async for invoice in stripe_list_items(page):
        late = _paid_late(dict(invoice), paused)
        if late is None or not _invoice_subscription_id(dict(invoice)):
            continue
        period_start, paid_at = late
        notes = await stripe_call(
            stripe.CreditNote.list_async, invoice=invoice["id"], limit=100
        )
        credited = any(
            (note.get("metadata") or {}).get(RESTARTED_FROM_INVOICE_KEY)
            == invoice["id"]
            for note in notes.data
        )
        rows.append(
            {
                "customer_id": invoice["customer"],
                "subscription_id": _invoice_subscription_id(dict(invoice)),
                "invoice_id": invoice["id"],
                "billing_reason": invoice.get("billing_reason"),
                "period_start": _iso(period_start),
                "paid_at": _iso(paid_at),
                "days_late": round((paid_at - period_start) / 86400, 1),
                "amount_paid": invoice.get("amount_paid"),
                "currency": invoice.get("currency"),
                "credited": credited,
            }
        )
    restarted = await _restarted_invoices([row["invoice_id"] for row in rows])
    return [{**row, "restarted": row["invoice_id"] in restarted} for row in rows]


async def _paused_invoices(since: datetime) -> set[str]:
    """Invoices whose failed payment was recorded as pausing the plan, from
    a little before ``since``, since a pause comes before the payment."""
    from prisma.models import SubscriptionBillingRestart

    rows = await SubscriptionBillingRestart.prisma().find_many(
        where={"pausedAt": {"gte": since - timedelta(days=60)}}
    )
    return {row.stripeInvoiceId for row in rows}


async def _restarted_invoices(invoice_ids: list[str]) -> set[str]:
    """Invoices whose restart finished: credited AND re-anchored. A credit
    without a restart needs a person."""
    from prisma.models import SubscriptionBillingRestart

    if not invoice_ids:
        return set()
    rows = await SubscriptionBillingRestart.prisma().find_many(
        where={"stripeInvoiceId": {"in": invoice_ids}, "restartedAt": {"not": None}}
    )
    return {row.stripeInvoiceId for row in rows}


async def _app_tiers(customer_ids: set[str]) -> dict[str, tuple[str, str]]:
    from prisma.models import User

    if not customer_ids:
        return {}
    users = await User.prisma().find_many(
        where={"stripeCustomerId": {"in": sorted(customer_ids)}}
    )
    return {
        user.stripeCustomerId: (user.id, str(user.subscriptionTier or ""))
        for user in users
        if user.stripeCustomerId
    }


def _iso(timestamp: int | None) -> str:
    if timestamp is None:
        return ""
    return datetime.fromtimestamp(timestamp, UTC).isoformat()


def _since(value: str) -> datetime:
    """An ISO date or time; one without a timezone is read as UTC."""
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List accounts hurt by a failed payment, read-only."
    )
    parser.add_argument("report", choices=["blocked", "late"])
    parser.add_argument(
        "--since",
        type=_since,
        default=datetime(2026, 9, 1, tzinfo=UTC),
        help="late: only invoices created on or after this date (default 2026-09-01)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    sys.exit(asyncio.run(main(args.report, args.since)))
