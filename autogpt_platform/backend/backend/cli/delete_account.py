import asyncio
import logging
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from backend.data.account_deletion.models import DeletionReport


@click.command(name="delete-account")
@click.argument("account")
@click.option(
    "--lock",
    is_flag=True,
    help="Stop the account and start the 14-day undo window, instead of "
    "deleting it.",
)
@click.option(
    "--unlock",
    is_flag=True,
    help="Let a locked account back in. Cancelled subscriptions stay cancelled.",
)
@click.option("--apply", is_flag=True, help="Make the change. Without it, dry run.")
@click.option("--yes", is_flag=True, help="With --apply, skip the confirmation.")
@click.option(
    "--force",
    is_flag=True,
    help="Delete without waiting out the undo window (or without a lock), "
    "or finish a deletion whose account rows are gone already.",
)
def delete_account_command(
    account: str, lock: bool, unlock: bool, apply: bool, yes: bool, force: bool
):
    """Delete an account and everything it holds, for a deletion request.

    ACCOUNT is the user ID or the account's email. Two steps, 14 days apart:

    --lock --apply stops the account: it bans sign-in and ends its sessions,
    opts it out of marketing, cancels its Stripe subscriptions and deletes
    its schedules and stops its runs. --unlock --apply lets it back in.

    Without --lock, the full deletion: it repeats those, hands the agents
    other accounts use (with their listings) to a "deleted user" account,
    deletes the rest of its marketplace presence, its files in every bucket,
    its AutoPilot memory, its PostHog person and MailerLite subscriber, and
    its rows and auth identity last. The credit ledger and Stripe's billing
    records stay. It refuses until 14 days after the lock, and an ID with no
    account left, unless --force.

    Dry run by default, with counts only. Idempotent: a failed --apply stops
    with the account in place, and running it again resumes. Run the dry run
    again afterwards to confirm nothing is left (it reports the account not
    found, with every count 0). The runbook is
    docs/engineering/account-deletion.md.
    """
    if lock and unlock:
        raise click.UsageError("Pass --lock or --unlock, not both.")
    if force and (lock or unlock):
        raise click.UsageError("--force only applies to the full deletion.")
    # Keep Prisma and client chatter out of the report.
    logging.disable(logging.INFO)
    mode = "lock" if lock else "unlock" if unlock else "delete"
    asyncio.run(_run(account, mode=mode, apply=apply, yes=yes, force=force))


_CONFIRM = {
    "lock": "Lock account {}: ban sign-in, cancel its subscriptions, delete "
    "its schedules?",
    "unlock": "Unlock account {}?",
    "delete": "Delete account {} for good?",
}


async def _run(account: str, *, mode: str, apply: bool, yes: bool, force: bool):
    from backend.data.account_deletion.models import AccountDeletionBlocked
    from backend.data.db import connect, disconnect

    await connect()
    try:
        user_id, email = await _resolve(account)
        plan = await _call(mode, user_id, email, dry_run=True, force=force)
        _report(plan)
        if not apply:
            click.echo("\nDry run: nothing was changed. Re-run with --apply.")
            return
        if plan.blockers:
            raise click.ClickException("Blocked: resolve the blockers above first.")
        if not yes:
            click.confirm("\n" + _CONFIRM[mode].format(user_id), abort=True)
        try:
            result = await _call(mode, user_id, email, dry_run=False, force=force)
        except AccountDeletionBlocked as e:
            _report(e.report)
            raise click.ClickException("Blocked since the dry run; nothing changed.")
        click.echo("")
        _report(result)
        _finish(result, mode)
    finally:
        await disconnect()


async def _call(
    mode: str, user_id: str, email: str | None, *, dry_run: bool, force: bool
) -> "DeletionReport":
    from backend.data.account_deletion import routine

    if mode == "lock":
        return await routine.lock_account(user_id, dry_run=dry_run, email=email)
    if mode == "unlock":
        return await routine.unlock_account(user_id, dry_run=dry_run)
    return await routine.delete_account(
        user_id, dry_run=dry_run, email=email, force=force
    )


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


_DONE = {
    "lock": "\nLocked. Run the full deletion after the date above, or --unlock "
    "to let the account back in.",
    "unlock": "\nUnlocked. Subscriptions stay cancelled and deleted schedules "
    "are not restored: tell the requester.",
    "delete": "\nDone. Run the dry run again to confirm nothing is left.",
}


def _finish(report: "DeletionReport", mode: str) -> None:
    """Fail the command unless every step went through, so a script never
    reads a partial run as done."""
    if not report.complete:
        raise click.ClickException(
            "The run did not finish (see the failed step above and the log); "
            "run it again to resume."
        )
    click.echo(_DONE[mode])
