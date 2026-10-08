import asyncio
import logging
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from backend.notifications.role_backfill import RolePlan, RoleRecord

# Every account with a role on record: the pick the wizard kept, or the copy
# in its business understanding. Only the role is read from the
# understanding, which can hold whole brain dump transcripts.
_ROLE_RECORDS = """
    SELECT u."id" AS user_id, u."email", u."timezone",
           u."marketingOptOutAt" AS marketing_opt_out_at,
           u."stripeCustomerId" AS stripe_customer_id,
           u."marketingExcludedCountry" AS excluded_country,
           o."role" AS choice, o."roleOther" AS other,
           c."data"->'business'->>'user_role' AS understanding_role
    FROM {schema_prefix}"User" u
    LEFT JOIN {schema_prefix}"UserOnboarding" o ON o."userId" = u."id"
    LEFT JOIN {schema_prefix}"CoPilotUnderstanding" c ON c."userId" = u."id"
    WHERE o."role" IS NOT NULL
       OR c."data"->'business'->>'user_role' IS NOT NULL
"""


@click.command(name="onboarding-role-backfill")
@click.option("--apply", is_flag=True, help="Write the changes. Without it, dry run.")
@click.option("--yes", is_flag=True, help="With --apply, skip the confirmation.")
def onboarding_role_backfill_command(apply: bool, yes: bool):
    """Give existing accounts their onboarding role in PostHog and MailerLite.

    Reliable roles only: the pick the wizard kept, which the profile step has
    sent itself since SECRT-2852, or an exact option ID in the business
    understanding, the only value there that can only have come from the
    wizard. Any other value is Other's typed text or an AutoPilot rewrite, and
    is counted and skipped.

    PostHog gets onboarding_role and onboarding_role_other with $set_once, so
    a role it already holds stays. MailerLite gets role and role_other only on
    subscribers that have no role yet. Nobody is created there, and anyone who
    opted out of marketing or whom a signal places in Iran or Russia, their
    Stripe billing country included, is left out.

    Dry run by default, with counts only. Idempotent, so a repeated --apply
    resumes an interrupted one; run the dry run again afterwards to confirm
    nothing is left.
    """
    # Keep Prisma and client chatter out of the report.
    logging.disable(logging.INFO)
    asyncio.run(_run(apply=apply, yes=yes))


async def _run(*, apply: bool, yes: bool) -> None:
    from backend.cli.mailerlite_backfill import stripe_billing_countries
    from backend.data.db import connect, disconnect
    from backend.notifications import mailerlite
    from backend.notifications import mailerlite_field_backfill as field_backfill
    from backend.notifications import role_backfill
    from backend.notifications.mailerlite import MailerLiteNotConfigured
    from backend.util.posthog_client import get_posthog_client
    from backend.util.settings import Settings

    if not Settings().secrets.stripe_api_key:
        raise click.ClickException("STRIPE_API_KEY is not set.")
    try:
        current = await field_backfill.read_current()
    except MailerLiteNotConfigured as e:
        raise click.ClickException(str(e))
    billing_countries = await stripe_billing_countries()
    await connect()
    try:
        records = _with_billing_countries(await _records(), billing_countries)
    finally:
        await disconnect()

    plan = role_backfill.plan(records, current)
    _report(plan)

    if not apply:
        click.echo("\nDry run: nothing was written. Re-run with --apply to write.")
        return
    if not plan.posthog and not plan.mailerlite:
        click.echo("\nNothing to write.")
        return
    if get_posthog_client() is None:
        raise click.ClickException("POSTHOG_API_KEY is not set.")
    if not yes:
        click.confirm(
            f"\nSet the role on {len(plan.posthog)} PostHog persons and "
            f"{len(plan.mailerlite)} MailerLite subscribers?",
            abort=True,
        )
    await role_backfill.send_to_posthog(plan.posthog)
    click.echo(f"PostHog: {len(plan.posthog)} sent")
    if plan.mailerlite:
        await mailerlite.ensure_fields()
        ok, failed, skipped = await role_backfill.apply(plan.mailerlite, _progress)
        _finish(ok, failed, skipped)


async def _records() -> "list[RoleRecord]":
    from backend.data.db import query_raw_with_schema
    from backend.notifications.role_backfill import RoleRecord

    return await query_raw_with_schema(_ROLE_RECORDS, model=RoleRecord)


def _with_billing_countries(
    records: "list[RoleRecord]", billing_countries: dict[str, str]
) -> "list[RoleRecord]":
    return [
        record.model_copy(
            update={
                "billing_country": billing_countries.get(
                    record.stripe_customer_id or ""
                )
            }
        )
        for record in records
    ]


def _report(plan: "RolePlan") -> None:
    from backend.notifications.role_backfill import WRITE_INTERVAL_SECONDS

    roles = ", ".join(
        f"{label}={count}"
        for label, count in sorted(plan.roles.items(), key=lambda kv: -kv[1])
    )
    click.echo(f"Accounts with a role on record: {plan.accounts}")
    click.echo(f"  reliable, kept by the wizard: {plan.kept}")
    click.echo(f"  reliable, an exact option in the understanding: {plan.exact}")
    click.echo(f"  skipped, Other's text or an AutoPilot rewrite: {plan.skipped}")
    click.echo(f"Roles: {roles}")
    click.echo(f"\nPostHog: {len(plan.posthog)} persons ($set_once)")
    click.echo(f"MailerLite: {len(plan.mailerlite)} subscribers to fill in")
    click.echo(f"  already have a role (kept): {plan.already_set}")
    click.echo(f"  not subscribers (never created): {plan.not_subscribers}")
    click.echo(f"  opted out of marketing (skipped): {plan.opted_out}")
    click.echo(f"  placed in Iran or Russia (skipped): {plan.excluded_country}")
    minutes = len(plan.mailerlite) * WRITE_INTERVAL_SECONDS / 60
    click.echo(f"About {minutes:.0f} minutes at one subscriber every two seconds")


def _progress(done: int, total: int) -> None:
    click.echo(f"MailerLite: {done}/{total} written")


def _finish(ok: int, failed: int, skipped: int) -> None:
    """Report the run, and fail it when anyone was not written, so a Job or
    script never reads a partial run as done. A rerun retries only what is
    left."""
    click.echo(f"MailerLite: {ok} ok, {failed} failed, {skipped} no longer needed")
    if failed:
        raise click.ClickException(
            f"{failed} MailerLite subscribers were not written (the log says "
            "why); run again to retry them"
        )
    click.echo("Run the dry run again to confirm nothing is left.")
