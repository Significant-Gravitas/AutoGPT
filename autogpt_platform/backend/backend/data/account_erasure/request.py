"""The hand-off from closing an account (SECRT-2877) to erasing it: the
`AccountDeletionRequest` row.

Closing the account writes the row (`close_account`), with `eraseAfter` 14
days on; until then the owner can sign in and keep the account, which deletes
the row. The erasure processes every row whose `completedAt` is still null
and whose `eraseAfter` has passed (`pending_user_ids`), and for each one, as
`account_deletion/service.py` asks:
- claims it (`claim_request`) before any step, so a Keep can no longer win;
- reruns `stop_user_work` and removes the memory crons (`work.py`);
- sends the account-deleted email if closing could not (`confirmationSentAt`
  null), before the address goes with the account, then records it;
- deletes the auth identity with the `User` row (`database.py`);
- sets `completedAt` once everything is gone (`complete_request`).

`startedAt` is never set back to null. An erasure that stopped halfway leaves
the row claimed and unfinished, and the next run (which does not filter on
`startedAt`) resumes it.

A request made by email to privacy@ has no row yet: `close_for_email_request`
closes the account the way Settings does, with source EMAIL. Its 14 days
start then, as for any other request.
"""

import asyncio
from datetime import UTC, datetime

from prisma.enums import AccountDeletionSource
from prisma.models import AccountDeletionRequest

from backend.api.features.account_deletion.service import delete_account
from backend.data.account_deletion import mark_confirmation_sent
from backend.data.account_erasure.models import StepReport, StepStatus, Subject
from backend.data.db import query_raw_with_schema
from backend.data.user import get_user_by_id
from backend.notifications.auth import send_account_deleted_email

CONFIRMATION_TIMEOUT_SECONDS = 60.0


async def pending_user_ids(*, now: datetime | None = None) -> list[str]:
    """The accounts whose 14 days to keep them are over and that are not
    erased yet, oldest first. One whose erasure started and stopped is
    included, so the next run resumes it."""
    rows = await AccountDeletionRequest.prisma().find_many(
        where={"completedAt": None, "eraseAfter": {"lte": now or datetime.now(UTC)}},
        order={"eraseAfter": "asc"},
    )
    return [row.userId for row in rows]


async def claim_request(user_id: str) -> str | None:
    """Claim the account's request for the erasure, in one short statement of
    its own, before any step runs: from then on Keep refuses (it deletes only
    an unclaimed row). Returns the request's ID, or None when there is nothing
    to claim: the owner kept the account, the 14 days are not over, or the
    erasure finished. A claim that an earlier, unfinished run made keeps its
    first date. The database's clock decides here, read in UTC as the
    columns are stored (they have no time zone); Keep (`reopen_account`)
    compares with the app server's clock instead. They can't both win
    whatever the skew: each statement re-checks its condition on the locked
    row, Keep deletes only an unclaimed row and the claim needs the row.
    No lock is held once it returns."""
    rows = await query_raw_with_schema(
        'UPDATE {schema_prefix}"AccountDeletionRequest" '
        'SET "startedAt" = COALESCE("startedAt", now() AT TIME ZONE \'UTC\') '
        'WHERE "userId" = $1 AND "completedAt" IS NULL '
        "AND \"eraseAfter\" <= now() AT TIME ZONE 'UTC' "
        'RETURNING "id"',
        user_id,
    )
    return rows[0]["id"] if rows else None


async def close_for_email_request(user_id: str) -> datetime:
    """Close the account for a request made by email, exactly as the Settings
    button does (`delete_account`): Stripe cancelled, apps disconnected, work
    stopped, email switched off, sessions ended, listings taken down, the
    request row written and the confirmation sent. Returns when the 14 days
    to keep the account end; the erasure follows after that. A no-op for an
    account already closed. Raises `PlanCancellationError`, with the account
    still open, when Stripe refuses the cancel."""
    user = await get_user_by_id(user_id)
    return await delete_account(user_id, user.email, AccountDeletionSource.EMAIL)


async def send_confirmation(subject: Subject, *, dry_run: bool) -> StepReport:
    """Send the account-deleted email closing could not send, while the
    address still exists, worded for the request's source. By now the 14 days
    are over, so the email says the account is being erased rather than
    naming a last day to keep it. A failure fails the step, so the erasure
    stops before the address goes and the next run tries again."""
    if subject.confirmation_sent or subject.request_source is None:
        return StepReport(
            name="confirmation",
            status=StepStatus.WOULD_DO if dry_run else StepStatus.DONE,
            counts={"emails": 0},
            note=(
                "sent when the account was closed"
                if subject.confirmation_sent
                else "closing the account sends it"
            ),
        )
    if not subject.email or subject.erase_after is None:
        # Not a skip: the address goes only with the rows, which an earlier
        # run deleted, so there is no one left to send it to.
        return StepReport(
            name="confirmation",
            status=StepStatus.WOULD_DO if dry_run else StepStatus.DONE,
            counts={"emails": 0},
            note="no address left to send it to; the account's rows are gone",
        )
    if not dry_run:
        await asyncio.wait_for(
            send_account_deleted_email(
                subject.email,
                subject.request_source,
                subject.erase_after,
                # Claimed, so the 14 days are over whatever this pod's clock
                # says: the email must not offer a Keep that is refused.
                now=max(datetime.now(UTC), subject.erase_after),
            ),
            timeout=CONFIRMATION_TIMEOUT_SECONDS,
        )
        await mark_confirmation_sent(subject.user_id)
    return StepReport(
        name="confirmation",
        status=StepStatus.WOULD_DO if dry_run else StepStatus.DONE,
        counts={"emails": 1},
    )


async def complete_request(subject: Subject, *, dry_run: bool) -> StepReport:
    """Mark the claimed request done, once: a rerun keeps the first date.
    Only the row this erasure claimed is marked, never a newer request of the
    same user ID."""
    if dry_run:
        count = int(not subject.request_completed)
    elif subject.request_id is None:
        raise RuntimeError("the request was not claimed")
    else:
        count = await AccountDeletionRequest.prisma().update_many(
            where={
                "id": subject.request_id,
                "startedAt": {"not": None},
                "completedAt": None,
            },
            data={"completedAt": datetime.now(UTC)},
        )
    return StepReport(
        name="request",
        status=StepStatus.WOULD_DO if dry_run else StepStatus.DONE,
        counts={"completed": count},
    )
