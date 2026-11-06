"""List accounts hurt by a failed payment, read-only (SECRT-2873).

Two lists, printed as CSV:

``blocked``  subscriptions that are ``past_due`` or ``unpaid`` with a bill the
             customer can still pay, and the app tier of their account. A
             ``NO_TIER`` account here is locked out of paid features until it
             pays.
``late``     paid renewal or first invoices (``subscription_cycle`` or
             ``subscription_create``) whose payment landed more than
             ``LATE_PAYMENT_AFTER`` after the period they opened had begun.
             Before the billing-period restart shipped, such a customer kept
             the old renewal date, so they paid for the days without access.
             ``restarted`` says whether the restart's credit note exists.

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
from datetime import UTC, datetime

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
    from backend.data.subscription_billing_restart import current_period_start

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


async def _late(since: datetime) -> list[dict]:
    import stripe

    from backend.data.credit import _invoice_subscription_id
    from backend.data.stripe_client import stripe_call, stripe_list_items
    from backend.data.subscription_billing_restart import (
        RESTARTED_FROM_INVOICE_KEY,
        late_payment,
    )

    rows = []
    page = await stripe_call(
        stripe.Invoice.list_async,
        status="paid",
        created={"gte": int(since.timestamp())},
        limit=100,
    )
    async for invoice in stripe_list_items(page):
        late = late_payment(dict(invoice))
        if late is None or not _invoice_subscription_id(dict(invoice)):
            continue
        notes = await stripe_call(
            stripe.CreditNote.list_async, invoice=invoice["id"], limit=100
        )
        restarted = any(
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
                "period_start": _iso(late.period_start),
                "paid_at": _iso(late.paid_at),
                "days_late": round(late.seconds_late / 86400, 1),
                "amount_paid": invoice.get("amount_paid"),
                "currency": invoice.get("currency"),
                "restarted": restarted,
            }
        )
    return rows


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


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("report", choices=["blocked", "late"])
    parser.add_argument(
        "--since",
        type=lambda value: datetime.fromisoformat(value).replace(tzinfo=UTC),
        default=datetime(2026, 9, 1, tzinfo=UTC),
        help="late: only invoices created on or after this date (default 2026-09-01)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    sys.exit(asyncio.run(main(args.report, args.since)))
