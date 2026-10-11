"""Closing an account that its owner asked to delete.

The ``AccountDeletionRequest`` row closes the account: backend auth rejects
its user, and the data erasure routine (SECRT-2854) later processes every row
whose ``completedAt`` is still null.
"""

from prisma.enums import AccountDeletionSource
from prisma.errors import UniqueViolationError
from prisma.models import AccountDeletionRequest

from backend.data.db import transaction

# Better Auth's banReason for a closed account. Its sign-in then fails with the
# bannedUserMessage the frontend configures for the admin plugin.
ACCOUNT_DELETED_BAN_REASON = "account_deleted"


async def get_deletion_request(user_id: str) -> AccountDeletionRequest | None:
    return await AccountDeletionRequest.prisma().find_unique(where={"userId": user_id})


async def is_account_closed(user_id: str) -> bool:
    return await get_deletion_request(user_id) is not None


async def close_account(user_id: str, source: AccountDeletionSource) -> bool:
    """Record the deletion request, ban the sign-in and end every session.

    All three happen in one transaction. Returns ``False``, having changed
    nothing, when the account was already closed (including by a concurrent
    call that won the race); ``True`` when this call closed it.
    """
    try:
        async with transaction() as tx:
            await tx.accountdeletionrequest.create(
                data={"userId": user_id, "source": source}
            )
            await tx.authuser.update_many(
                where={"id": user_id},
                data={
                    "banned": True,
                    "banReason": ACCOUNT_DELETED_BAN_REASON,
                    "banExpires": None,
                },
            )
            await tx.authsession.delete_many(where={"userId": user_id})
    except UniqueViolationError:
        return False
    return True
