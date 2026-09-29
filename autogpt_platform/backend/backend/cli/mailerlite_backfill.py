import asyncio
import logging
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from backend.notifications.mailerlite_backfill import (
        Customer,
        PlannedChange,
        Subscription,
    )
    from backend.notifications.mailerlite_field_backfill import FieldPlan, Person

_ACCOUNT_PAGE = 5000


@click.command(name="mailerlite-backfill")
@click.option("--apply", is_flag=True, help="Write the changes. Without it, dry run.")
@click.option("--yes", is_flag=True, help="With --apply, skip the confirmation.")
@click.option(
    "--fields-only",
    is_flag=True,
    help="Only the status and date fields; leave group membership alone.",
)
def mailerlite_backfill_command(apply: bool, yes: bool, fields_only: bool):
    """Put existing accounts where the live MailerLite code would have.

    Groups: paying customers join the changelog unless they are in the
    onboarding tour; churned customers leave it. When
    MAILERLITE_TRIAL_GROUP_ID is set, the trial group ends up holding exactly
    the customers on a trial that is not set to cancel.

    Fields: every account, with or without a Stripe customer, gets its
    subscription_status and dates. An account MailerLite does not have yet is
    created as a subscriber.

    Dry run by default: prints counts and one pseudonymised line per customer,
    and writes nothing. Idempotent, so a partial or repeated --apply is safe,
    and running it again resumes an interrupted one.
    """
    # Keep Prisma and client chatter out of the report.
    logging.disable(logging.INFO)
    asyncio.run(_run(apply=apply, yes=yes, fields_only=fields_only))


@click.command(name="mailerlite-fields")
@click.option("--apply", is_flag=True, help="Create the missing fields.")
def mailerlite_fields_command(apply: bool):
    """Create the MailerLite subscriber fields the backend writes, if missing.

    The notification service also creates them before its first field write,
    so this only sets them up ahead of time. Idempotent.
    """
    logging.disable(logging.INFO)
    asyncio.run(_fields(apply=apply))


async def _fields(*, apply: bool) -> None:
    from backend.notifications import mailerlite
    from backend.notifications.mailerlite import FIELD_TYPES, MailerLiteNotConfigured

    try:
        mailerlite._require_token()
        existing = await mailerlite.read_fields()
    except MailerLiteNotConfigured as e:
        raise click.ClickException(str(e))
    for field, kind in FIELD_TYPES.items():
        held = existing.get(field.value)
        if held is None:
            state = "missing"
        elif held != kind:
            state = f"exists as {held}, expected {kind}"
        else:
            state = "ok"
        click.echo(f"{field.value:<28} {kind:<5} {state}")
    if not apply:
        click.echo("\nDry run: nothing was created. Re-run with --apply to create.")
        return
    created = await mailerlite.ensure_fields()
    click.echo(f"\nCreated {len(created)} fields.")


async def _run(*, apply: bool, yes: bool, fields_only: bool) -> None:
    from backend.data.db import connect, disconnect
    from backend.notifications import mailerlite, mailerlite_backfill
    from backend.notifications import mailerlite_field_backfill as field_backfill
    from backend.notifications.mailerlite import MailerLiteNotConfigured
    from backend.notifications.mailerlite_backfill import CHANGES
    from backend.util.settings import Settings

    if not Settings().secrets.stripe_api_key:
        raise click.ClickException("STRIPE_API_KEY is not set.")
    try:
        audience = None if fields_only else await mailerlite_backfill.read_audience()
        current = await field_backfill.read_current()
    except MailerLiteNotConfigured as e:
        raise click.ClickException(str(e))
    subscriptions = await _stripe_subscriptions()
    await connect()
    try:
        people = await _people(subscriptions)
    finally:
        await disconnect()

    changes: "list[PlannedChange]" = []
    if audience is not None:
        customers = [_customer(p) for p in people if p.subscriptions]
        changes = mailerlite_backfill.plan(customers, audience)
        _report(changes, len(subscriptions) - len(customers))
    fields = field_backfill.plan(people, current)
    _report_fields(fields, len(people))

    if not apply:
        click.echo("\nDry run: nothing was written. Re-run with --apply to write.")
        return
    due = sum(d in CHANGES for c in changes for d in c.decisions)
    if not due and not fields.changes:
        click.echo("\nNothing to write.")
        return
    if not yes:
        click.confirm(
            f"\nWrite {due} group changes and {len(fields.changes)} field updates?",
            abort=True,
        )
    if due and audience is not None:
        result = await mailerlite_backfill.apply(changes, audience)
        for decision in CHANGES:
            click.echo(
                f"{decision.value}: {result.succeeded[decision]} ok, "
                f"{result.failed[decision]} failed"
            )
    if fields.changes:
        await mailerlite.ensure_fields()
        ok, failed = await field_backfill.apply(fields.changes, _progress)
        click.echo(f"fields: {ok} ok, {failed} failed")


async def _stripe_subscriptions() -> "dict[str, list[Subscription]]":
    """Every Stripe subscription, by customer ID."""
    import stripe

    from backend.data.stripe_client import stripe_call, stripe_list_items
    from backend.notifications.mailerlite_backfill import Subscription
    from backend.util.settings import Settings

    stripe.api_key = Settings().secrets.stripe_api_key
    subscriptions: dict[str, list[Subscription]] = {}
    page = await stripe_call(stripe.Subscription.list_async, status="all", limit=100)
    async for sub in stripe_list_items(page):
        # Not expanded, so this is the customer ID.
        subscriptions.setdefault(str(sub.customer), []).append(
            Subscription(
                id=str(sub.id),
                status=str(sub.status),
                cancel_at_period_end=bool(sub.get("cancel_at_period_end")),
                from_trial=bool((sub.get("metadata") or {}).get("trial_enrollment_id")),
                start_date=sub.get("start_date"),
                trial_start=sub.get("trial_start"),
                trial_end=sub.get("trial_end"),
                canceled_at=sub.get("canceled_at"),
                ended_at=sub.get("ended_at"),
            )
        )
    return subscriptions


async def _people(subscriptions: "dict[str, list[Subscription]]") -> "list[Person]":
    """Every account, paged, with its Stripe subscriptions. The account's
    email is used, as the live handlers do, never Stripe's."""
    import prisma.models

    from backend.notifications.mailerlite_field_backfill import Person

    converted = {
        t.stripeSubscriptionId
        for t in await prisma.models.SubscriptionTrial.prisma().find_many(
            where={"convertedAt": {"not": None}}
        )
        if t.stripeSubscriptionId
    }
    people: list[Person] = []
    cursor: str | None = None
    while True:
        page = await prisma.models.User.prisma().find_many(
            take=_ACCOUNT_PAGE,
            order={"id": "asc"},
            **({"cursor": {"id": cursor}, "skip": 1} if cursor else {}),
        )
        for user in page:
            subs = subscriptions.get(user.stripeCustomerId or "", [])
            people.append(
                Person(
                    user_id=user.id,
                    email=user.email,
                    created_at=user.createdAt,
                    subscriptions=[
                        s.model_copy(update={"converted": s.id in converted})
                        for s in subs
                    ],
                )
            )
        if len(page) < _ACCOUNT_PAGE:
            return people
        cursor = page[-1].id


def _customer(person: "Person") -> "Customer":
    from backend.notifications.mailerlite_backfill import Customer

    return Customer(
        user_id=person.user_id,
        email=person.email,
        subscriptions=person.subscriptions,
    )


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


def _report_fields(plan: "FieldPlan", accounts: int) -> None:
    from backend.notifications.mailerlite_backfill import (
        BATCH_SIZE,
        UPSERT_BATCH_INTERVAL_SECONDS,
    )

    click.echo(f"\nFields for {accounts} accounts")
    for status, count in plan.statuses.items():
        click.echo(f"  {status.value}: {count}")
    click.echo(f"  invalid email (skipped): {plan.invalid}")
    new = sum(c.new for c in plan.changes)
    click.echo(
        f"{len(plan.changes)} to write: {new} new MailerLite subscribers, "
        f"{len(plan.changes) - new} existing ones updated"
    )
    batches = -(-len(plan.changes) // BATCH_SIZE)
    minutes = batches * UPSERT_BATCH_INTERVAL_SECONDS / 60
    click.echo(f"About {batches} batches, {minutes:.0f} minutes at the import limit")


def _progress(done: int, total: int) -> None:
    click.echo(f"fields: {done}/{total} written")
