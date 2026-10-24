import asyncio
import logging
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from backend.data.account_erasure.models import ErasureReport


@click.command(name="erase-account")
@click.argument("account", required=False)
@click.option(
    "--pending",
    is_flag=True,
    help="Erase every closed account not erased yet, instead of one.",
)
@click.option(
    "--close",
    is_flag=True,
    help="For a request made by email: close the account the way Settings "
    "does first, then erase it.",
)
@click.option(
    "--allow-banned",
    is_flag=True,
    help="With --close: close an account banned for another reason too, "
    "keeping that ban.",
)
@click.option("--apply", is_flag=True, help="Make the change. Without it, dry run.")
@click.option("--yes", is_flag=True, help="With --apply, skip the confirmation.")
def erase_account_command(
    account: str | None,
    pending: bool,
    close: bool,
    allow_banned: bool,
    apply: bool,
    yes: bool,
):
    """Erase a closed account and everything it holds (SECRT-2854).

    ACCOUNT is the user ID or the account's email. Only a closed account (one
    with a deletion request, from Settings, the paywall or --close) is
    erased. --pending erases every closed account not erased yet, as the
    periodic job does.

    --close is for a request made by email to privacy@: it closes the account
    exactly as the Settings button does (Stripe plan cancelled, apps
    disconnected, work and email stopped, sign-in banned, the confirmation
    email sent), then erases it. Like Settings, it refuses an account banned
    for another reason (abuse, say); --allow-banned closes it anyway and
    keeps that ban's reason and expiry.

    The erasure stops the account's work again, sends the confirmation if it
    has not gone out, cancels the subscriptions of the orgs that go with the
    account, hands the agents other accounts use to a "deleted user" account,
    deletes the rest of its marketplace presence, its files in every bucket,
    its AutoPilot memory, its PostHog person and MailerLite subscriber, its
    rows and auth identity last, and marks the request done. The credit
    ledger and Stripe's billing records stay.

    Dry run by default, with counts only. Idempotent: a failed --apply stops
    with the account in place, and running it again resumes. The runbook is
    docs/engineering/account-deletion.md.
    """
    if pending == bool(account):
        raise click.UsageError("Pass an ACCOUNT or --pending, not both.")
    if close and pending:
        raise click.UsageError("--close applies to one account.")
    if allow_banned and not close:
        raise click.UsageError("--allow-banned applies with --close.")
    # Keep Prisma and client chatter out of the report.
    logging.disable(logging.INFO)
    if pending:
        asyncio.run(_with_db(_run_pending(apply=apply, yes=yes)))
    else:
        asyncio.run(
            _with_db(
                _run_one(
                    account or "",
                    close=close,
                    allow_banned=allow_banned,
                    apply=apply,
                    yes=yes,
                )
            )
        )


async def _with_db(work):
    from backend.data.db import connect, disconnect

    await connect()
    try:
        await work
    finally:
        await disconnect()


async def _run_one(
    account: str, *, close: bool, allow_banned: bool, apply: bool, yes: bool
):
    from backend.data.account_deletion import is_account_closed
    from backend.data.account_erasure.models import AccountErasureBlocked
    from backend.data.account_erasure.routine import erase_account

    user_id, email = await _resolve(account)
    closing = close and not await is_account_closed(user_id)
    plan = await erase_account(
        user_id,
        dry_run=True,
        email=email,
        closing=closing,
        allow_banned=allow_banned,
    )
    _report(plan)
    if closing:
        click.echo(_CLOSE_FIRST)
    elif close:
        click.echo("\nThe account is closed already; --close changes nothing.")
    if not apply:
        click.echo("\nDry run: nothing was changed. Re-run with --apply.")
        return
    if plan.blockers:
        raise click.ClickException("Blocked: resolve the blockers above first.")
    if not yes:
        click.confirm(f"\nErase account {user_id} for good?", abort=True)
    if closing:
        await _close(user_id, allow_banned=allow_banned)
    try:
        result = await erase_account(user_id, dry_run=False, email=email)
    except AccountErasureBlocked as e:
        _report(e.report)
        raise click.ClickException("Blocked since the dry run; nothing erased.")
    click.echo("")
    _report(result)
    _finish([result])


_CLOSE_FIRST = (
    "\nThe account is open: --apply first closes it as a request made by "
    "email, the way Settings does (Stripe plan cancelled, apps disconnected, "
    "work and email stopped, sign-in banned, confirmation email sent)."
)


async def _close(user_id: str, *, allow_banned: bool) -> None:
    from backend.api.features.account_deletion.service import PlanCancellationError
    from backend.data.account_erasure.request import (
        AccountBannedError,
        close_for_email_request,
    )

    try:
        await close_for_email_request(user_id, allow_banned=allow_banned)
    except AccountBannedError:
        raise click.ClickException(
            "The account was banned for another reason since the dry run, so "
            "nothing was closed or erased. Pass --allow-banned to close it "
            "anyway, keeping that ban."
        )
    except PlanCancellationError:
        raise click.ClickException(
            "Stripe refused or timed out on the plan cancel, so the account "
            "is still open and nothing was erased. Run it again."
        )
    click.echo("\nClosed the account.")


async def _run_pending(*, apply: bool, yes: bool):
    from backend.data.account_erasure.sweep import erase_each, erase_pending

    plans = await erase_pending(dry_run=True)
    if not plans:
        click.echo("No closed account is waiting to be erased.")
        return
    for plan in plans:
        _report(plan)
        click.echo("")
    ready = [plan.user_id for plan in plans if not plan.blockers]
    click.echo(f"{len(ready)} of {len(plans)} closed accounts can be erased.")
    if not apply:
        click.echo("Dry run: nothing was changed. Re-run with --apply.")
        return
    if not ready:
        raise click.ClickException("Every one is blocked: resolve the blockers.")
    if not yes:
        click.confirm(f"\nErase {len(ready)} accounts for good?", abort=True)
    results = await erase_each(ready, dry_run=False)
    for result in results:
        click.echo("")
        _report(result)
    _finish(results)


async def _resolve(account: str) -> tuple[str, str | None]:
    """The user ID, and the email when that is what was given. An email is
    looked up on the platform account, then on the auth identity, so a
    half-finished erasure can still be found by it."""
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
        "No account has that email. If it was erased already, pass its user "
        "ID to finish or confirm the erasure."
    )


def _report(report: "ErasureReport") -> None:
    mode = "Dry run" if report.dry_run else "Applied"
    click.echo(f"{mode} for account {report.user_id}")
    if not report.account_found:
        click.echo("  no platform account or auth identity (already erased?)")
    for blocker in report.blockers:
        click.echo(f"BLOCKED {blocker.name}: {blocker.count}")
        click.echo(f"  {blocker.resolution}")
    for step in report.steps:
        counts = ", ".join(f"{key}={value}" for key, value in step.counts.items())
        line = f"  {step.name:<17} {step.status.value:<9} {counts}"
        click.echo(f"{line}  ({step.note})" if step.note else line)


def _finish(reports: list["ErasureReport"]) -> None:
    """Fail the command unless every step of every erasure went through, so
    a script never reads a partial run as done."""
    unfinished = [report.user_id for report in reports if not report.complete]
    if unfinished:
        raise click.ClickException(
            f"{len(unfinished)} erasure(s) did not finish (see the failed step "
            "above and the log); run it again to resume."
        )
    click.echo("\nDone. Run the dry run again to confirm nothing is left.")
