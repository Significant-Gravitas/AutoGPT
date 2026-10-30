"""Keeping an account its owner closed (SECRT-2877).

For 14 days after the request (``eraseAfter``) the account is closed but not
erased. Signing in during that time shows when it will be erased, with "Keep
my account" and "Sign out". Keeping it cancels the request, so the erasure
never runs, and lets the owner back in at once on this worker and within the
closed-account cache TTL on the others.

What closing stopped stays stopped: the plan (they subscribe again), the
connected apps, API keys and app grants (they reconnect), schedules and
triggers, share links, Marketplace listings and reviews, and product email
(switched back on in settings). A person banned for another reason can't sign
in, so they can't keep the account.
"""

import logging

from autogpt_libs.auth import forget_account_state

from backend.api.features.account_deletion.models import AccountDeletionResponse
from backend.data.account_deletion import (
    ReopenOutcome,
    get_deletion_request,
    reopen_account,
)

logger = logging.getLogger(__name__)


class TooLateToRestoreError(Exception):
    """The 14 days are over: the erasure may already be running."""


async def get_deletion_status(user_id: str) -> AccountDeletionResponse:
    """Read from the database, never the cache. An open answer also drops
    this worker's cached one: right after the owner keeps the account,
    another worker may still hold "closed" and send them back here."""
    request = await get_deletion_request(user_id)
    if request is None:
        forget_account_state(user_id)
        return AccountDeletionResponse(status="open")
    return AccountDeletionResponse(status="scheduled", erase_after=request.eraseAfter)


async def restore_account(user_id: str) -> None:
    """Cancel the deletion request. A no-op for an account that is open.

    Raises TooLateToRestoreError, having changed nothing, once the erasure
    date has passed.
    """
    outcome = await reopen_account(user_id)
    if outcome is ReopenOutcome.TOO_LATE:
        logger.info(f"Account deletion: user {user_id} asked to keep it too late")
        raise TooLateToRestoreError()
    # Open either way, so this worker lets the owner back in at once.
    forget_account_state(user_id)
    if outcome is ReopenOutcome.REOPENED:
        logger.info(f"Account deletion cancelled: user {user_id} kept the account")
