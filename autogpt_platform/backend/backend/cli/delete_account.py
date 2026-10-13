import asyncio
import logging
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from backend.data.account_deletion.models import DeletionReport


@click.command(name="delete-account")
@click.argument("account")
@click.option("--apply", is_flag=True, help="Delete. Without it, dry run.")
@click.option("--yes", is_flag=True, help="With --apply, skip the confirmation.")
@click.option(
    "--remove-shared-agents",
    is_flag=True,
    help="Take this account's agents out of other accounts' libraries, which "
    "otherwise blocks the deletion. Their runs of those agents go too.",
)
def delete_account_command(
    account: str, apply: bool, yes: bool, remove_shared_agents: bool
):
    """Delete an account and everything it holds, for a deletion request.

    ACCOUNT is the user ID or the account's email. Stops the account first
    (sign-in, schedules, runs, Stripe subscriptions, integrations), then
    deletes its marketplace listings and reviews, its files in every bucket,
    its AutoPilot memory, its PostHog person and MailerLite subscriber, and
    its rows and auth identity last. Stripe's billing records stay.

    Dry run by default, with counts only. Idempotent: a failed --apply stops
    with the account in place, and running it again resumes. Run the dry run
    again afterwards to confirm nothing is left. The runbook is
    docs/engineering/account-deletion.md.
    """
    # Keep Prisma and client chatter out of the report.
    logging.disable(logging.INFO)
    asyncio.run(
        _run(account, apply=apply, yes=yes, remove_shared_agents=remove_shared_agents)
    )


async def _run(
    account: str, *, apply: bool, yes: bool, remove_shared_agents: bool
) -> None:
    from backend.data.account_deletion.models import AccountDeletionBlocked
    from backend.data.account_deletion.routine import delete_account
    from backend.data.db import connect, disconnect

    await connect()
    try:
        user_id, email = await _resolve(account)
        plan = await delete_account(
            user_id,
            dry_run=True,
            email=email,
            remove_shared_agents=remove_shared_agents,
        )
        _report(plan)
        if not apply:
            click.echo("\nDry run: nothing was deleted. Re-run with --apply.")
            return
        if plan.blockers:
            raise click.ClickException("Blocked: resolve the blockers above first.")
        if not yes:
            click.confirm(f"\nDelete account {user_id} for good?", abort=True)
        try:
            result = await delete_account(
                user_id,
                dry_run=False,
                email=email,
                remove_shared_agents=remove_shared_agents,
            )
        except AccountDeletionBlocked as e:
            _report(e.report)
            raise click.ClickException("Blocked since the dry run; nothing deleted.")
        click.echo("")
        _report(result)
        _finish(result)
    finally:
        await disconnect()


async def _resolve(account: str) -> tuple[str, str | None]:
    """The user ID, and the email when that is what was given. An email is
    looked up on the platform account, then on the auth identity, so a
    half-finished deletion can still be found by it."""
    from prisma.models import AuthUser, User

    if "@" not in account:
        return account, None
    by_email = {"email": {"equals": account, "mode": "insensitive"}}
    user = await User.prisma().find_first(where=by_email)
    if user:
        return user.id, account
    identity = await AuthUser.prisma().find_first(where=by_email)
    if identity:
        return identity.id, account
    raise click.ClickException(
        "No account has that email. If it was deleted already, pass its user "
        "ID to finish or confirm the deletion."
    )


def _report(report: "DeletionReport") -> None:
    mode = "Dry run" if report.dry_run else "Applied"
    click.echo(f"{mode} for account {report.user_id}")
    if not report.account_found:
        click.echo("  no platform account or auth identity (already deleted?)")
    for blocker in report.blockers:
        click.echo(f"BLOCKED {blocker.name}: {blocker.count}")
        click.echo(f"  {blocker.resolution}")
    for step in report.steps:
        counts = ", ".join(f"{key}={value}" for key, value in step.counts.items())
        line = f"  {step.name:<14} {step.status.value:<9} {counts}"
        click.echo(f"{line}  ({step.note})" if step.note else line)


def _finish(report: "DeletionReport") -> None:
    """Fail the command unless everything went, so a script never reads a
    partial deletion as done."""
    if not report.complete:
        raise click.ClickException(
            "The deletion did not finish (see the failed step above and the "
            "log); run it again to resume."
        )
    click.echo("\nDone. Run the dry run again to confirm nothing is left.")
