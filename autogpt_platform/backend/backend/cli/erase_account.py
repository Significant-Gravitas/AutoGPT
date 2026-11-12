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
    "does. It is erased once its 14 days to keep it are over.",
)
@click.option(
    "--now",
    is_flag=True,
    help="With --close, for an owner who asked to be erased at once: no 14 "
    "days to keep it. Closes the account if it is open, then erases it in the "
    "same run.",
)
@click.option("--apply", is_flag=True, help="Make the change. Without it, dry run.")
@click.option("--yes", is_flag=True, help="With --apply, skip the confirmation.")
def erase_account_command(
    account: str | None, pending: bool, close: bool, now: bool, apply: bool, yes: bool
):
    """Erase a closed account and everything it holds (SECRT-2854).

    ACCOUNT is the user ID or the account's email. Only a closed account (one
    with a deletion request, from Settings, the paywall or --close) whose 14
    days to keep it are over is erased. --pending erases every such account
    not erased yet, as the periodic job does.

    --close is for a request made by email to privacy@: it only closes the
    account, exactly as the Settings button does (Stripe plan cancelled, apps
    disconnected, work and email stopped, the confirmation email sent). The
    owner then has 14 days to sign in and keep it, as from Settings.

    The erasure first claims the request, so the owner can no longer keep the
    account. It then stops the account's work again and removes its memory
    crons, sends the confirmation if it has not gone out, cancels the
    subscriptions of the orgs that go with the account, hands the agents
    other accounts use to a "deleted user" account, deletes the rest of its
    marketplace presence, its files in every bucket, its AutoPilot memory,
    its PostHog person and MailerLite subscriber, keeps a ban record if the
    account is banned, deletes its rows and auth identity last, and marks the
    request done. The credit ledger and Stripe's billing records stay.

    Dry run by default, with counts only. Idempotent: a failed --apply stops
    with the account in place, and running it again resumes. The runbook is
    docs/engineering/account-deletion.md.
    """
    if pending == bool(account):
        raise click.UsageError("Pass an ACCOUNT or --pending, not both.")
    if close and pending:
        raise click.UsageError("--close applies to one account.")
    if now and not close:
        raise click.UsageError("--now goes with --close.")
    _log_the_erasure_only()
    if pending:
        asyncio.run(_with_db(_run_pending(apply=apply, yes=yes)))
    elif close and now:
        asyncio.run(_with_db(_run_close_now(account or "", apply=apply, yes=yes)))
    elif close:
        asyncio.run(_with_db(_run_close(account or "", apply=apply, yes=yes)))
    else:
        asyncio.run(_with_db(_run_one(account or "", apply=apply, yes=yes)))


def _log_the_erasure_only() -> None:
    """The backend's logging, so each run leaves the erasure's own lines
    ("Account erasure of <user ID>: ...", IDs only) in the log, and only
    warnings from everything else, which keeps Prisma and client chatter out
    of the report."""
    from backend.util.logging import configure_logging

    configure_logging()
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("backend.data.account_erasure").setLevel(logging.INFO)


async def _with_db(work):
    from backend.data.db import connect, disconnect

    await connect()
    try:
        await work
    finally:
        await disconnect()


async def _run_one(account: str, *, apply: bool, yes: bool):
    from backend.data.account_erasure.lease import ErasureLeaseUnavailable
    from backend.data.account_erasure.models import AccountErasureBlocked
    from backend.data.account_erasure.routine import erase_account

    user_id, email = await _resolve(account)
    plan = await erase_account(user_id, dry_run=True, email=email)
    _report(plan)
    if not apply:
        click.echo("\nDry run: nothing was changed. Re-run with --apply.")
        return
    if plan.blockers:
        raise click.ClickException("Blocked: resolve the blockers above first.")
    if not yes:
        click.confirm(f"\nErase account {user_id} for good?", abort=True)
    try:
        result = await erase_account(user_id, dry_run=False, email=email)
    except AccountErasureBlocked as e:
        _report(e.report)
        raise click.ClickException("Blocked since the dry run; nothing erased.")
    except ErasureLeaseUnavailable:
        raise click.ClickException(
            "Redis can't be reached, so it is unknown whether another run is "
            "erasing this account. Nothing was changed; run it again."
        )
    click.echo("")
    _report(result)
    _finish([result])


async def _run_close(account: str, *, apply: bool, yes: bool):
    """Close the account for a request made by email; erase nothing."""
    from backend.api.features.account_deletion.service import PlanCancellationError
    from backend.data.account_deletion import get_deletion_request
    from backend.data.account_erasure.request import close_for_email_request

    user_id, _ = await _resolve(account)
    _refuse_a_system_id(user_id)
    if request := await get_deletion_request(user_id):
        click.echo(
            f"Account {user_id} is closed already; it is erased after "
            f"{request.eraseAfter:%Y-%m-%d %H:%M} UTC. --close changes nothing."
        )
        return
    click.echo(f"Account {user_id} is open.{_CLOSE_DOES}")
    if not apply:
        click.echo("\nDry run: nothing was changed. Re-run with --apply.")
        return
    if not yes:
        click.confirm(f"\nClose account {user_id}?", abort=True)
    try:
        erase_after = await close_for_email_request(user_id)
    except PlanCancellationError:
        raise click.ClickException(
            "Stripe refused or timed out on the plan cancel, so the account "
            "is still open. Run it again."
        )
    click.echo(
        f"\nClosed the account. Its owner can keep it until "
        f"{erase_after:%Y-%m-%d %H:%M} UTC; the job erases it after that."
    )


async def _run_close_now(account: str, *, apply: bool, yes: bool):
    """For an owner who asked to be erased at once: close the account if it
    is open, with no 14 days to keep it, then erase it in the same run. The
    dry run is checked before anything is closed, for blockers and for a step
    that would stop the erasure (one that fails, or a service that is not set
    up): closing first would leave the account closed, its owner unable to
    keep it, and not (fully) erased."""
    from backend.api.features.account_deletion.service import PlanCancellationError
    from backend.data.account_deletion import get_deletion_request
    from backend.data.account_erasure.request import (
        close_for_email_request,
        end_grace_now,
    )
    from backend.data.account_erasure.routine import erase_account, stops_an_apply

    user_id, email = await _resolve(account)
    _refuse_a_system_id(user_id)
    request = await get_deletion_request(user_id)
    if request is None:
        click.echo(
            f"Account {user_id} is open. --close --now --apply closes it as a "
            "request made by email, with no 14 days to keep it (the email says "
            "it is being erased), then erases it."
        )
    else:
        click.echo(
            f"Account {user_id} is closed; its 14 days end "
            f"{request.eraseAfter:%Y-%m-%d %H:%M} UTC. --close --now --apply "
            "ends them now and erases it."
        )
    plan = await erase_account(user_id, dry_run=True, email=email)
    _report(plan)
    if any(b.name not in _CLOSE_NOW_CLEARS for b in plan.blockers):
        raise click.ClickException(
            "Blocked: resolve the blockers above first. Nothing was closed or "
            "erased."
        )
    if stopping := [step.name for step in plan.steps if stops_an_apply(step)]:
        raise click.ClickException(
            f"The erasure would stop at: {', '.join(stopping)}. Fix that first "
            "(see the notes above). Nothing was closed or erased."
        )
    if not apply:
        click.echo("\nDry run: nothing was changed. Re-run with --apply.")
        return
    if not yes:
        click.confirm(f"\nClose and erase account {user_id} now, for good?", abort=True)
    if request is None:
        try:
            await close_for_email_request(user_id, erase_now=True)
        except PlanCancellationError:
            raise click.ClickException(
                "Stripe refused or timed out on the plan cancel, so the account "
                "is still open. Run it again."
            )
        click.echo("\nClosed the account, with no 14 days to keep it.")
    await end_grace_now(user_id)
    await _run_one(account, apply=True, yes=True)


# What --close --now itself clears: it closes the account and ends its 14 days.
_CLOSE_NOW_CLEARS = ("not_closed", "not_due")


def _refuse_a_system_id(user_id: str) -> None:
    """Closing runs no preflight, so it checks the ID itself: never close the
    placeholder the shared agents go to, or an ID no account can have."""
    from backend.data.account_erasure.preflight import invalid_id, system_account

    if refused := invalid_id(user_id) or system_account(user_id):
        raise click.ClickException(refused[0].resolution)


_CLOSE_DOES = (
    " --close --apply closes it as a request made by email, the way Settings "
    "does: Stripe plan cancelled, apps disconnected, work and email stopped, "
    "confirmation email sent. The owner then has 14 days to keep it."
)


async def _run_pending(*, apply: bool, yes: bool):
    from backend.data.account_erasure.sweep import (
        erase_each,
        erase_pending,
        purge_ban_records,
    )

    plans = await erase_pending(dry_run=True)
    if not plans:
        click.echo("No closed account is waiting to be erased.")
        if apply:
            click.echo(f"Purged {await purge_ban_records()} expired ban record(s).")
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
        # The purge is the sweep's too, and does not wait on the blockers.
        click.echo(f"Purged {await purge_ban_records()} expired ban record(s).")
        raise click.ClickException("Every one is blocked: resolve the blockers.")
    if not yes:
        click.confirm(f"\nErase {len(ready)} accounts for good?", abort=True)
    results = await erase_each(ready, dry_run=False)
    for result in results:
        click.echo("")
        _report(result)
    click.echo(f"\nPurged {await purge_ban_records()} expired ban record(s).")
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
    if report.not_claimed:
        click.echo(f"  nothing ran: {report.not_claimed}")
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
    from backend.data.account_erasure.routine import ERASED_ALREADY

    finished = [r for r in reports if r.not_claimed == ERASED_ALREADY]
    if any(report.not_claimed and report not in finished for report in reports):
        raise click.ClickException(
            "At least one account was not erased: its owner kept it, or another "
            "run is erasing it (see above)."
        )
    unfinished = [
        report.user_id
        for report in reports
        if not report.complete and report not in finished
    ]
    if unfinished:
        raise click.ClickException(
            f"{len(unfinished)} erasure(s) did not finish (see the failed step "
            "above and the log); run it again to resume."
        )
    click.echo("\nDone. Run the dry run again to confirm nothing is left.")
