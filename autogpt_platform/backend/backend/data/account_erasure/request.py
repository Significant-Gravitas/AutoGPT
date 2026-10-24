"""The hand-off from closing an account (SECRT-2877) to erasing it: the
`AccountDeletionRequest` row.

Closing the account writes the row (`close_account`). The erasure processes
every row whose `completedAt` is still null (`pending_user_ids`), and for
each one, as `account_deletion/service.py` asks:
- reruns `stop_user_work` (`work.py`);
- sends the account-deleted email if closing could not (`confirmationSentAt`
  null), before the address goes with the account, then records it;
- deletes the auth identity with the `User` row (`database.py`);
- sets `completedAt` once everything is gone.

A request made by email to privacy@ has no row yet: `close_for_email_request`
closes the account the way Settings does, with source EMAIL.
"""

import asyncio
from datetime import UTC, datetime, timedelta

from prisma.enums import AccountDeletionSource
from prisma.models import AccountDeletionRequest, AuthUser

from backend.api.features.account_deletion.service import delete_account
from backend.data.account_deletion import mark_confirmation_sent
from backend.data.account_erasure.models import StepReport, StepStatus, Subject
from backend.data.account_erasure.preflight import banned_for_another_reason
from backend.data.user import get_user_by_id
from backend.notifications.auth import send_account_deleted_email

# A request this fresh is left for the next run: closing sends the
# confirmation just after it writes the row, and an erasure in between would
# send a second one.
PENDING_GRACE = timedelta(minutes=15)
CONFIRMATION_TIMEOUT_SECONDS = 60.0


async def pending_user_ids(
    *, grace: timedelta = PENDING_GRACE, now: datetime | None = None
) -> list[str]:
    """The accounts closed at least `grace` ago and not erased yet, oldest
    first."""
    cutoff = (now or datetime.now(UTC)) - grace
    rows = await AccountDeletionRequest.prisma().find_many(
        where={"completedAt": None, "requestedAt": {"lte": cutoff}},
        order={"requestedAt": "asc"},
    )
    return [row.userId for row in rows]


class AccountBannedError(Exception):
    """The account is banned for another reason, and the operator did not
    allow closing it anyway. Nothing was changed."""


async def close_for_email_request(user_id: str, *, allow_banned: bool = False) -> None:
    """Close the account for a request made by email, exactly as the Settings
    button does (`delete_account`): Stripe cancelled, apps disconnected, work
    stopped, email switched off, sign-in banned, sessions ended, listings
    taken down, the request row written and the confirmation sent. A no-op
    for an account already closed. Raises `PlanCancellationError`, with the
    account still open, when Stripe refuses the cancel.

    Like the Settings route, it refuses an account banned for another reason
    (`AccountBannedError`, before Stripe or anything else is touched) unless
    `allow_banned`; `close_account` then keeps that ban's reason and expiry."""
    if not allow_banned and banned_for_another_reason(
        await AuthUser.prisma().find_unique(where={"id": user_id})
    ):
        raise AccountBannedError()
    user = await get_user_by_id(user_id)
    await delete_account(user_id, user.email, AccountDeletionSource.EMAIL)


async def send_confirmation(subject: Subject, *, dry_run: bool) -> StepReport:
    """Send the account-deleted email closing could not send, while the
    address still exists, worded for the request's source. A failure fails the step, so the erasure stops
    before the address goes and the next run tries again."""
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
    if not subject.email:
        return StepReport(
            name="confirmation",
            status=StepStatus.SKIPPED,
            note="no address left to send it to",
        )
    if not dry_run:
        await asyncio.wait_for(
            send_account_deleted_email(subject.email, subject.request_source),
            timeout=CONFIRMATION_TIMEOUT_SECONDS,
        )
        await mark_confirmation_sent(subject.user_id)
    return StepReport(
        name="confirmation",
        status=StepStatus.WOULD_DO if dry_run else StepStatus.DONE,
        counts={"emails": 1},
    )


async def complete_request(subject: Subject, *, dry_run: bool) -> StepReport:
    """Mark the request done, once: a rerun keeps the first date."""
    if dry_run:
        count = int(not subject.request_completed)
    else:
        count = await AccountDeletionRequest.prisma().update_many(
            where={"userId": subject.user_id, "completedAt": None},
            data={"completedAt": datetime.now(UTC)},
        )
    return StepReport(
        name="request",
        status=StepStatus.WOULD_DO if dry_run else StepStatus.DONE,
        counts={"completed": count},
    )
