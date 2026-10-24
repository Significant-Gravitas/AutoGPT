"""Closing an account that its owner asked to delete.

The ``AccountDeletionRequest`` row closes the account: backend auth rejects
its user, and the data erasure routine (SECRT-2854) later processes every row
whose ``completedAt`` is still null.
"""

from datetime import datetime, timezone

from prisma.enums import AccountDeletionSource
from prisma.errors import UniqueViolationError
from prisma.models import AccountDeletionRequest

from backend.data.db import transaction

# Better Auth's banReason for a closed account. Its sign-in then fails as
# closed (the frontend's closed-account plugin); any other ban as banned.
ACCOUNT_DELETED_BAN_REASON = "account_deleted"


async def get_deletion_request(user_id: str) -> AccountDeletionRequest | None:
    return await AccountDeletionRequest.prisma().find_unique(where={"userId": user_id})


async def is_account_closed(user_id: str) -> bool:
    return await get_deletion_request(user_id) is not None


async def close_account(user_id: str, source: AccountDeletionSource) -> bool:
    """Record the deletion request, ban the sign-in, end every session and
    take the user's Marketplace listings down.

    A ban for another reason (an abuse ban an operator chose to close with
    ``erase-account --close --allow-banned``; the Settings route refuses such
    an account) keeps its reason and expiry, so it is not mistaken for a
    deletion.

    All of it happens in one transaction. Listings are soft-deleted, pending
    ones too so they can't be approved later, and the erasure removes them
    with the User. Returns ``False``, having changed nothing, when the account
    was already closed (including by a concurrent call that won the race);
    ``True`` when this call closed it.
    """
    try:
        async with transaction() as tx:
            await tx.accountdeletionrequest.create(
                data={"userId": user_id, "source": source}
            )
            await tx.authuser.update_many(
                where={
                    "id": user_id,
                    "OR": [
                        {"banned": None},
                        {"banned": False},
                        {"banReason": ACCOUNT_DELETED_BAN_REASON},
                    ],
                },
                data={
                    "banned": True,
                    "banReason": ACCOUNT_DELETED_BAN_REASON,
                    "banExpires": None,
                },
            )
            await tx.authsession.delete_many(where={"userId": user_id})
            await tx.storelisting.update_many(
                where={"owningUserId": user_id, "isDeleted": False},
                data={"isDeleted": True},
            )
            await tx.skilllisting.update_many(
                where={"owningUserId": user_id, "isDeleted": False},
                data={"isDeleted": True},
            )
    except UniqueViolationError:
        return False
    return True


async def mark_confirmation_sent(user_id: str) -> None:
    """Record that the account-deleted email went out. The erasure routine
    sends it for a request still unmarked, before it erases the address."""
    await AccountDeletionRequest.prisma().update(
        where={"userId": user_id},
        data={"confirmationSentAt": datetime.now(timezone.utc)},
    )
