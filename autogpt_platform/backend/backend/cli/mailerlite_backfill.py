import asyncio
import logging
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from backend.notifications.mailerlite_backfill import Customer, PlannedChange

_ACCOUNT_LOOKUP_CHUNK = 500


@click.command(name="mailerlite-backfill")
@click.option("--apply", is_flag=True, help="Write the changes. Without it, dry run.")
@click.option("--yes", is_flag=True, help="With --apply, skip the confirmation.")
def mailerlite_backfill_command(apply: bool, yes: bool):
    """Put existing Stripe customers in the MailerLite groups the live
    lifecycle code would have put them in.

    Paying customers join the changelog unless they are in the onboarding
    tour; churned customers leave it. When MAILERLITE_TRIAL_GROUP_ID is set,
    the trial group ends up holding exactly the customers on a trial that is
    not set to cancel. Dry run by default: prints counts
    and one pseudonymised line per customer, and writes nothing. Idempotent,
    so a partial or repeated --apply is safe.
    """
    # Keep Prisma and client chatter out of the report.
    logging.disable(logging.INFO)
    asyncio.run(_run(apply=apply, yes=yes))


async def _run(*, apply: bool, yes: bool) -> None:
    from backend.data.db import connect, disconnect
    from backend.notifications import mailerlite_backfill
    from backend.notifications.mailerlite import MailerLiteNotConfigured
    from backend.notifications.mailerlite_backfill import CHANGES
    from backend.util.settings import Settings

    if not Settings().secrets.stripe_api_key:
        raise click.ClickException("STRIPE_API_KEY is not set.")
    try:
        audience = await mailerlite_backfill.read_audience()
    except MailerLiteNotConfigured as e:
        raise click.ClickException(str(e))
    await connect()
    try:
        customers, unmatched = await _customers()
    finally:
        await disconnect()

    changes = mailerlite_backfill.plan(customers, audience)
    _report(changes, unmatched)

    if not apply:
        click.echo("\nDry run: nothing was written. Re-run with --apply to write.")
        return
    due = sum(d in CHANGES for c in changes for d in c.decisions)
    if not due:
        click.echo("\nNothing to write.")
        return
    if not yes:
        click.confirm(f"\nWrite {due} MailerLite changes?", abort=True)
    result = await mailerlite_backfill.apply(changes, audience)
    for decision in CHANGES:
        click.echo(
            f"{decision.value}: {result.succeeded[decision]} ok, "
            f"{result.failed[decision]} failed"
        )


async def _customers() -> "tuple[list[Customer], int]":
    """Every Stripe customer with a subscription, matched to their account.
    The account's email is used, as the live handlers do, never Stripe's."""
    import prisma.models
    import stripe

    from backend.data.stripe_client import stripe_call, stripe_list_items
    from backend.notifications.mailerlite_backfill import Customer, Subscription
    from backend.util.settings import Settings

    stripe.api_key = Settings().secrets.stripe_api_key
    subscriptions: dict[str, list[Subscription]] = {}
    page = await stripe_call(stripe.Subscription.list_async, status="all", limit=100)
    async for sub in stripe_list_items(page):
        # Not expanded, so this is the customer ID.
        subscriptions.setdefault(str(sub.customer), []).append(
            Subscription(
                status=str(sub.status),
                cancel_at_period_end=bool(sub.get("cancel_at_period_end")),
                from_trial=bool((sub.get("metadata") or {}).get("trial_enrollment_id")),
            )
        )

    ids = list(subscriptions)
    users = [
        user
        for start in range(0, len(ids), _ACCOUNT_LOOKUP_CHUNK)
        for user in await prisma.models.User.prisma().find_many(
            where={
                "stripeCustomerId": {"in": ids[start : start + _ACCOUNT_LOOKUP_CHUNK]}
            }
        )
    ]
    customers = [
        Customer(
            user_id=u.id,
            email=u.email,
            subscriptions=subscriptions[u.stripeCustomerId],
        )
        for u in users
        if u.stripeCustomerId
    ]
    return customers, len(ids) - len(customers)


def _report(changes: "list[PlannedChange]", unmatched: int) -> None:
    from collections import Counter

    from backend.notifications.mailerlite import _pseudonym

    for change in changes:
        click.echo(
            f"{_pseudonym(change.customer.email)}  {change.standing.value:<16}  "
            + ", ".join(d.value for d in change.decisions)
        )
    counts = Counter(d.value for c in changes for d in c.decisions)
    click.echo(f"\n{len(changes)} customers with an account")
    click.echo(f"{unmatched} Stripe customers with no account (skipped)")
    for name, count in sorted(counts.items()):
        click.echo(f"  {name}: {count}")
