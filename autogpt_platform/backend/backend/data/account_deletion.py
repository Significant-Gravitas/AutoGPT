"""Closing an account that its owner asked to delete, and reopening it.

The ``AccountDeletionRequest`` row closes the account: backend auth rejects
its user. For ``GRACE_PERIOD`` the owner can change their mind by signing in
and keeping the account, which deletes the row (``reopen_account``). After
``eraseAfter`` the data erasure routine (SECRT-2854) processes every row whose
``completedAt`` is still null.

Closing does not ban the sign-in: the owner must be able to sign in to keep
the account. A ban for another reason is left exactly as it was, so a banned
person can close their account but never sign in to reopen it.
"""

from datetime import datetime, timedelta, timezone
from enum import Enum

from prisma import Prisma
from prisma.enums import AccountDeletionSource
from prisma.errors import UniqueViolationError
from prisma.models import AccountDeletionRequest

from backend.data.db import transaction

# How long the owner has to change their mind before the data is erased.
GRACE_PERIOD = timedelta(days=14)

# The product-email limit a reopened account gets back: closing sets it to 0,
# and the settings page has no control for it.
REOPENED_DAILY_EMAIL_LIMIT = 3


class ReopenOutcome(Enum):
    REOPENED = "reopened"
    # Nothing to reopen: the account was never closed, or was reopened already.
    NOT_CLOSED = "not_closed"
    # The grace period is over, or the erasure has started or finished.
    TOO_LATE = "too_late"


async def get_deletion_request(user_id: str) -> AccountDeletionRequest | None:
    return await AccountDeletionRequest.prisma().find_unique(where={"userId": user_id})


async def is_account_closed(user_id: str) -> bool:
    return await get_deletion_request(user_id) is not None


async def close_account(
    user_id: str, source: AccountDeletionSource
) -> AccountDeletionRequest | None:
    """Record the deletion request, end every session, take the user's
    Marketplace listings and reviews down, turn off their share links and
    unlink their chat platforms.

    All of it happens in one transaction. Listings are soft-deleted, pending
    ones too so they can't be approved later, and the erasure removes them
    with the User. Reopening the account brings none of it back. Returns
    ``None``, having changed nothing, when the account was already closed
    (including by a concurrent call that won the race); the new request when
    this call closed it.
    """
    now = datetime.now(timezone.utc)
    try:
        async with transaction() as tx:
            request = await tx.accountdeletionrequest.create(
                data={
                    "userId": user_id,
                    "source": source,
                    "requestedAt": now,
                    "eraseAfter": now + GRACE_PERIOD,
                }
            )
            await tx.authsession.delete_many(where={"userId": user_id})
            await _take_down_marketplace(tx, user_id)
            await _turn_off_share_links(tx, user_id)
            await _unlink_chat_platforms(tx, user_id)
    except UniqueViolationError:
        return None
    return request


async def reopen_account(user_id: str, now: datetime | None = None) -> ReopenOutcome:
    """Cancel the deletion request while the grace period lasts.

    The delete is conditional on ``eraseAfter`` still being ahead and on the
    erasure routine not having claimed the request (``startedAt``), in one
    statement: the routine runs on another server whose clock may differ, so
    the date alone can't keep the two apart.
    Everything closing stopped stays stopped: the plan, connected apps, API
    keys, schedules, share links and listings. Product email is limited to
    ``REOPENED_DAILY_EMAIL_LIMIT`` a day again, with every kind still off
    until the owner switches it back on.
    """
    now = now or datetime.now(timezone.utc)
    async with transaction() as tx:
        reopened = await tx.accountdeletionrequest.delete_many(
            where={
                "userId": user_id,
                "startedAt": None,
                "completedAt": None,
                "eraseAfter": {"gt": now},
            }
        )
        if reopened:
            await tx.user.update_many(
                where={"id": user_id},
                data={"maxEmailsPerDay": REOPENED_DAILY_EMAIL_LIMIT},
            )
    if reopened:
        return ReopenOutcome.REOPENED
    if await is_account_closed(user_id):
        return ReopenOutcome.TOO_LATE
    return ReopenOutcome.NOT_CLOSED


async def mark_confirmation_sent(user_id: str) -> None:
    """Record that the account-closed email went out. The erasure routine
    sends it for a request still unmarked, before it erases the address."""
    await AccountDeletionRequest.prisma().update(
        where={"userId": user_id},
        data={"confirmationSentAt": datetime.now(timezone.utc)},
    )


async def _take_down_marketplace(tx: Prisma, user_id: str) -> None:
    """The listings go (published workflows keep working for people who
    added them), and so do the reviews the user left on other listings. The
    creator page is hidden while the account is closed
    (``store.db.get_store_creator``)."""
    await tx.storelisting.update_many(
        where={"owningUserId": user_id, "isDeleted": False},
        data={"isDeleted": True},
    )
    await tx.skilllisting.update_many(
        where={"owningUserId": user_id, "isDeleted": False},
        data={"isDeleted": True},
    )
    await tx.storelistingreview.delete_many(where={"reviewByUserId": user_id})


async def _turn_off_share_links(tx: Prisma, user_id: str) -> None:
    """End the share of every chat and run the user shared, with their file
    allowlists and chat links, as un-sharing each one does."""
    await tx.sharedchatfile.delete_many(where={"Session": {"is": {"userId": user_id}}})
    await tx.sharedexecutionfile.delete_many(
        where={"Execution": {"is": {"userId": user_id}}}
    )
    await tx.chatlinkedshare.delete_many(
        where={
            "OR": [
                {"Session": {"is": {"userId": user_id}}},
                {"Execution": {"is": {"userId": user_id}}},
            ]
        }
    )
    await tx.chatsession.update_many(
        where={"userId": user_id, "isShared": True},
        data={
            "isShared": False,
            "shareToken": None,
            "sharedAt": None,
            "autoShareExecutions": False,
        },
    )
    await tx.agentgraphexecution.update_many(
        where={"userId": user_id, "isShared": True},
        data={
            "isShared": False,
            "shareToken": None,
            "sharedAt": None,
            "sharedVia": None,
        },
    )


async def _unlink_chat_platforms(tx: Prisma, user_id: str) -> None:
    """The Discord, Telegram and Slack servers and DMs linked to the account,
    so the bot stops answering there. ``stop_user_work`` already unlinks them
    one by one, early, before AutoPilot turns are cancelled; this makes sure
    none is left."""
    await tx.platformlink.delete_many(where={"userId": user_id})
    await tx.platformuserlink.delete_many(where={"userId": user_id})
