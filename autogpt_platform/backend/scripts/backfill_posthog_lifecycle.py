"""Backfill subscription status and lifecycle dates onto PostHog persons.

Runs the same sweep as the daily ``sync_posthog_lifecycles`` scheduler job
(``backend.data.posthog_lifecycle_sync.sync_all_posthog_lifecycles``): one
Stripe ``Subscription.list(status="all")`` pass mapped to users by customer
id, trials from the database, and one ``$set`` per user sent in batches.

A dry run (the default) reads Stripe and the database and prints how many
users land in each status; it sends nothing. ``--send`` sends the ``$set``s.
By default only users with a Stripe customer or a trial row are covered, as
in the daily job; ``--all-users`` adds everyone else (``signed``, with
``signup_at``).

Needs DATABASE_URL and STRIPE_API_KEY, and POSTHOG_API_KEY/POSTHOG_HOST for
``--send``, pointing at the same environment.

Usage::

    poetry run python scripts/backfill_posthog_lifecycle.py             # dry run
    poetry run python scripts/backfill_posthog_lifecycle.py --send
    poetry run python scripts/backfill_posthog_lifecycle.py --send --all-users
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys


async def main(send: bool, all_users: bool, batch_size: int) -> int:
    if not os.environ.get("DATABASE_URL"):
        raise SystemExit("DATABASE_URL must be set")

    # Imported lazily so --help works without the backend env being loaded.
    import stripe

    from backend.data.db import connect, disconnect
    from backend.data.posthog_lifecycle_sync import sync_all_posthog_lifecycles
    from backend.util.posthog_client import get_posthog_client
    from backend.util.settings import Settings

    secrets = Settings().secrets
    if not secrets.stripe_api_key:
        raise SystemExit("STRIPE_API_KEY must be set")
    stripe.api_key = secrets.stripe_api_key
    mode = (
        "live" if secrets.stripe_api_key.startswith(("sk_live", "rk_live")) else "test"
    )
    print(f"Stripe: {mode} mode")
    if send:
        if not secrets.posthog_api_key:
            raise SystemExit("POSTHOG_API_KEY must be set to send")
        print(f"PostHog: {secrets.posthog_host}")

    await connect()
    try:
        summary = await sync_all_posthog_lifecycles(
            dry_run=not send, all_users=all_users, batch_size=batch_size
        )
    finally:
        await disconnect()
        client = get_posthog_client()
        if send and client is not None:
            client.flush()

    print(format_summary(summary.model_dump()))
    return 1 if summary.aborted or summary.errors else 0


def format_summary(summary: dict) -> str:
    lines = [
        "DRY RUN, nothing sent" if summary["dry_run"] else "SENT",
        f"Stripe subscriptions read: {summary['stripe_subscriptions']}",
        f"Users mapped: {summary['users']}",
    ]
    if summary["aborted"]:
        lines.append(f"ABORTED: {summary['aborted']}")
    lines += [f"  {status}: {n}" for status, n in summary["status_counts"].items()]
    if not summary["dry_run"]:
        lines.append(f"$set sent: {summary['sent']}")
    lines.append(f"Errors: {summary['errors']}")
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill subscription status and lifecycle dates onto PostHog."
    )
    parser.add_argument(
        "--send",
        action="store_true",
        help="Send the $set to PostHog. Without it this is a dry run.",
    )
    parser.add_argument(
        "--all-users",
        action="store_true",
        help="Include users with no Stripe customer and no trial.",
    )
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    return args


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    sys.exit(asyncio.run(main(args.send, args.all_users, args.batch_size)))
