import asyncio
import logging
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from backend.notifications.posthog_audience_backfill import Account, AudiencePlan

_ACCOUNTS = """
    SELECT u."id" AS user_id, u."email", u."createdAt" AS created_at,
           u."timezone", u."stripeCustomerId" AS stripe_customer_id
    FROM {schema_prefix}"User" u
"""


@click.command(name="posthog-audience-backfill")
@click.option("--apply", is_flag=True, help="Send the properties. Without it, dry run.")
@click.option("--yes", is_flag=True, help="With --apply, skip the confirmation.")
def posthog_audience_backfill_command(apply: bool, yes: bool):
    """Give existing accounts the PostHog person properties that mirror
    MailerLite's segmentation (SECRT-2895).

    Every account gets email_type and signin_method. Every account with a
    Stripe Checkout Session also gets checkout_opened_at, country,
    country_code, country_source and exclude_de_at, worked out as
    mailerlite-checkout-backfill works them out: from the Stripe billing
    country, or else the browser's timezone. Nobody is left out, since PostHog
    is analytics, not marketing.

    checkout_opened_at is $set from the first Checkout Session. A timezone
    country and an exclude_de_at "no" are $set_once, so what a live checkout
    recorded (its IP country, a German or Austrian signal) stays.

    Dry run by default, with counts only. A run that PostHog didn't take in
    full, at queueing or upload, fails after reporting; a repeated --apply
    sends the same values, so running it again is harmless.
    """
    # Keep Prisma and client chatter out of the report.
    logging.disable(logging.INFO)
    asyncio.run(_run(apply=apply, yes=yes))


async def _run(*, apply: bool, yes: bool) -> None:
    import stripe

    from backend.cli.mailerlite_backfill import (
        checkout_sessions,
        first_opens,
        signin_providers_by_account,
        stripe_billing_countries,
    )
    from backend.data.db import connect, disconnect
    from backend.notifications import posthog_audience_backfill as backfill
    from backend.util.settings import Settings

    settings = Settings()
    if not settings.secrets.stripe_api_key:
        raise click.ClickException("STRIPE_API_KEY is not set.")
    if apply and not settings.secrets.posthog_api_key:
        raise click.ClickException("POSTHOG_API_KEY is not set.")
    stripe.api_key = settings.secrets.stripe_api_key
    # Listed once, for the first opens and the billing countries alike.
    sessions = await checkout_sessions()
    billing_countries = await stripe_billing_countries(sessions)
    await connect()
    try:
        accounts = await _accounts()
        providers = await signin_providers_by_account()
    finally:
        await disconnect()

    plan = backfill.plan(accounts, providers, first_opens(sessions), billing_countries)
    _report(plan)

    if not apply:
        click.echo("\nDry run: nothing was sent. Re-run with --apply to send.")
        return
    if not plan.updates:
        click.echo("\nNothing to send.")
        return
    if not yes:
        click.confirm(
            f"\nSend the properties of {len(plan.updates)} accounts to PostHog?",
            abort=True,
        )
    _finish(len(plan.updates), await backfill.send(plan.updates))


async def _accounts() -> "list[Account]":
    from backend.data.db import query_raw_with_schema
    from backend.notifications.posthog_audience_backfill import Account

    return await query_raw_with_schema(_ACCOUNTS, model=Account)


def _report(plan: "AudiencePlan") -> None:
    click.echo(f"Accounts: {len(plan.updates)}")
    click.echo(f"  email_type: {_counts(plan.email_types)}")
    click.echo(f"  signin_method: {_counts(plan.signin_methods)}")
    click.echo(f"Checkout openers (a Stripe Checkout Session): {plan.openers}")
    click.echo(f"  country_source: {_counts(plan.country_sources)}")
    click.echo(f"  exclude_de_at yes: {plan.exclude_de_at}")
    click.echo(
        "These are the values the run sends. A timezone country and an "
        'exclude_de_at "no" are $set_once: they land only where the person '
        "holds none yet."
    )
    click.echo(f"\nPostHog: {len(plan.updates)} persons")


def _finish(total: int, failed: int) -> None:
    """Report the run, and fail it when PostHog didn't take everyone, so a Job
    or script never reads a partial run as done."""
    click.echo(f"PostHog: {total - failed} sent, {failed} failed")
    if failed:
        raise click.ClickException(
            f"{failed} accounts were not sent (the log says why); run again to "
            "retry them, since a run sends the same values every time"
        )


def _counts(counts: dict[str, int]) -> str:
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{value}={count}" for value, count in ordered) or "-"
