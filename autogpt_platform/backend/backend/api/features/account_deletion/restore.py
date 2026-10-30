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
from datetime import datetime, timezone

from autogpt_libs.auth import forget_account_state

from backend.api.features.account_deletion.analytics import tenure_bucket
from backend.api.features.account_deletion.models import AccountDeletionResponse
from backend.api.features.store.cache import forget_creator_page
from backend.data.account_deletion import (
    ReopenOutcome,
    get_deletion_request,
    reopen_account,
)
from backend.data.user import forget_refusal, get_user_by_id
from backend.util.posthog_client import capture_anonymous
from backend.util.posthog_events import PostHogEvent

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
    # The same test reopen_account makes, so the page never offers a keep
    # the server would refuse, whatever the browser's clock says.
    too_late = (
        request.startedAt is not None
        or request.completedAt is not None
        or request.eraseAfter <= datetime.now(timezone.utc)
    )
    return AccountDeletionResponse(
        status="too_late" if too_late else "scheduled",
        erase_after=request.eraseAfter,
    )


async def restore_account(user_id: str) -> None:
    """Cancel the deletion request. A no-op for an account that is open.

    Raises TooLateToRestoreError, having changed nothing, once the erasure
    date has passed.
    """
    request = await get_deletion_request(user_id)
    outcome = await reopen_account(user_id)
    if outcome is ReopenOutcome.TOO_LATE:
        logger.info(f"Account deletion: user {user_id} asked to keep it too late")
        raise TooLateToRestoreError()
    # Open either way, so this worker lets the owner back in at once, and
    # background work and the creator page come back without waiting out a
    # cache.
    forget_account_state(user_id)
    forget_refusal(user_id)
    if outcome is not ReopenOutcome.REOPENED:
        return
    logger.info(f"Account deletion cancelled: user {user_id} kept the account")
    await forget_creator_page(user_id)
    if request is not None:
        await _count_kept(user_id, request.eraseAfter)


async def _count_kept(user_id: str, erase_after: datetime) -> None:
    """One anonymous count, best effort: the account is open already."""
    try:
        user = await get_user_by_id(user_id)
        days_left = (erase_after - datetime.now(timezone.utc)).days
        capture_anonymous(
            PostHogEvent.ACCOUNT_DELETION_KEPT,
            {
                "tenure_bucket": tenure_bucket(user.created_at),
                "days_left": max(0, days_left),
            },
        )
    except Exception as e:
        logger.warning(
            f"Could not count user {user_id} keeping the account: {type(e).__name__}"
        )
