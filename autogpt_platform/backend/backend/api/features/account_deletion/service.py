"""Self-serve account deletion (SECRT-2877): close an account its owner asked
to delete. Its data is erased 14 days later (SECRT-2854); until then the
owner can sign in and keep it (``restore.py``).

``delete_account`` runs, in this order:

1. Load the user and check the typed email (trimmed, in any case).
2. Return at once when the account is already closed: a repeat call cancels,
   revokes and sends nothing, and records no second count. The person may be
   banned for another reason: deleting their account is their right, and the
   ban is left as it is.
3. Read the plan for the completed count, before Stripe is touched: the
   cancel's webhooks move the tier, and can land while this still runs.
4. Cancel every Stripe subscription now, with no invoice, and switch off
   auto top-up so work that outlives the deletion can't charge the saved
   card. A Stripe error or timeout stops everything here: the account stays
   open (some subscriptions may already be cancelled) and the person is asked
   to try again.
5. Disconnect the apps they connected, revoke the API keys and OAuth grants
   AutoGPT issued for them, stop their schedules, triggers, AutoPilot turns
   and runs, and switch off their product and marketing email.
6. Close the account (``close_account``): the deletion request row with the
   date erasure may start (14 days on), the end of every session, the
   Marketplace listings and reviews taken down, the share links turned off
   and the chat platforms unlinked, in one transaction. Sign-in is not
   banned: signing in is how the owner keeps the account. This worker
   rejects the user at once; the others within their cache TTL.
7. Send the confirmation email (with the last day to keep the account) and
   record one anonymous count.

Every step before the close must succeed or raise, so the account never
closes with a plan still billing. Two are best effort by design:
``stop_user_work`` never raises, and connected apps are handled per
credential: each is deleted from AutoGPT the way the Disconnect button deletes
it, then revoked at its provider, and one that fails or stalls is logged
without holding the account open, since a retry would meet the same refusal;
the erasure deletes whatever is still stored. Revoking at the provider is best
effort anyway: some providers (Notion, Todoist, Airtable, ...) have no revoke
endpoint, so their grant lasts until the user removes it or it expires.
Nothing is closed until step 6, so the person can simply try again, and each
step is safe to repeat:

- the Stripe cancel lists only subscriptions Stripe can still cancel, so a
  retry cancels just what the first attempt missed, and switching off auto
  top-up writes the same setting again;
- ``revoke_connected_apps`` works from the credentials still stored: the ones
  already deleted are gone, and the managed-credential cleanup logs, rather
  than raises, a resource that is already gone;
- ``revoke_issued_access`` revokes only keys and grants not revoked yet;
- ``stop_user_work`` lists what is still scheduled or running, and never
  raises;
- ``stop_account_emails`` writes the same settings again, and an earlier
  marketing opt-out keeps its date and source.

The steps after the close are best effort: the account is closed, so the call
succeeds even when the email cannot be sent, and it stops waiting for the email
after ``CONFIRMATION_EMAIL_TIMEOUT_SECONDS``. A sent email is recorded on the
request row (``confirmationSentAt``); one that failed is left unmarked for the
erasure routine. When two calls race, the one whose ``close_account`` loses
returns without sending anything; the winner sends the email and records the
count.

Hand-off to data erasure (SECRT-2854): the AccountDeletionRequest row is the
whole hand-off. That routine processes every row whose ``eraseAfter`` has
passed and whose ``startedAt`` and ``completedAt`` are still null, claiming it
by setting ``startedAt`` before it erases anything, so there is nothing to
enqueue here; a row deleted by reopening the account is simply gone. It
should first rerun ``stop_user_work``, as a schedule or trigger whose removal
failed here is not retried (``is_user_refused`` keeps them from starting work
meanwhile). It owns removing the memory crons closing leaves registered so
they resume if the account is kept: ``delete_nightly_batch_schedule``,
``delete_community_rebuild_schedule`` and ``delete_skill_learning_schedule``,
for the account and for each of its experts. It must also delete the user's
auth identities with the User row, or the auth-identity monitor re-creates
the User. A request whose ``confirmationSentAt`` is still null got no email:
by then the 14 days are over, so the routine words its own.

Neither the typed email nor the account's email is ever logged.
"""

import asyncio
import logging
from datetime import datetime, timezone

import stripe
from autogpt_libs.auth import remember_account_closed
from prisma.enums import AccountDeletionSource
from prisma.models import AccountDeletionRequest

from backend.api.features.account_deletion.analytics import plan_state
from backend.api.features.account_deletion.connected_apps import revoke_connected_apps
from backend.api.features.account_deletion.email_opt_out import stop_account_emails
from backend.api.features.account_deletion.issued_access import revoke_issued_access
from backend.api.features.account_deletion.scheduled_work import stop_user_work
from backend.api.features.store.cache import forget_store_listings
from backend.data.account_deletion import (
    GRACE_PERIOD,
    close_account,
    get_deletion_request,
    mark_confirmation_sent,
)
from backend.data.credit import cancel_all_subscriptions_now, set_auto_top_up
from backend.data.model import AutoTopUpConfig
from backend.data.user import forget_refusal, get_user_by_id
from backend.notifications.auth import send_account_deleted_email
from backend.util.posthog_client import capture_anonymous
from backend.util.posthog_events import PostHogEvent

logger = logging.getLogger(__name__)

# The account is already closed; the request must answer before the frontend
# proxy gives up on it.
CONFIRMATION_EMAIL_TIMEOUT_SECONDS = 15.0

# Where the flow was opened, as the completed count reports it. Not "source":
# that is the base property every event carries.
_SURFACES = {
    AccountDeletionSource.SETTINGS: "settings",
    AccountDeletionSource.PAYWALL: "paywall",
}


class EmailMismatchError(Exception):
    """The typed confirmation email isn't the account's email."""


class PlanCancellationError(Exception):
    """Stripe refused or timed out on the cancel. The account is still open,
    though some of its subscriptions may already be cancelled."""


async def delete_account(
    user_id: str, confirm_email: str, source: AccountDeletionSource
) -> datetime:
    """Close the account, in the order the module docstring gives, and return
    when its data may be erased: until then its owner can keep it.

    Raises EmailMismatchError, having changed nothing, for a different email,
    and PlanCancellationError when Stripe refuses or times out on the cancel:
    the account stays open, and a retry cancels whatever is left. Any other
    error before the close also leaves the account open, so a retry is safe.
    """
    user = await get_user_by_id(user_id)
    _check_email(user.email, confirm_email)
    existing = await get_deletion_request(user_id)
    if existing is not None:
        return existing.eraseAfter
    plan = plan_state(user.subscription_tier)
    await _cancel_plan(user_id)
    await set_auto_top_up(user_id, AutoTopUpConfig(threshold=0, amount=0))
    await _revoke_and_stop(user_id)
    request = await _close(user_id, source)
    if request is None:
        return await _winners_erase_after(user_id)
    await _send_confirmation(user_id, user.email, source, request.eraseAfter)
    capture_anonymous(
        PostHogEvent.ACCOUNT_DELETION_COMPLETED,
        {"plan_state": plan, "surface": _SURFACES[source]},
    )
    return request.eraseAfter


def _check_email(account_email: str, confirm_email: str) -> None:
    typed = confirm_email.strip().lower()
    if not typed or typed != account_email.strip().lower():
        raise EmailMismatchError()


async def _cancel_plan(user_id: str) -> None:
    try:
        cancelled = await cancel_all_subscriptions_now(user_id)
    except stripe.StripeError as e:
        logger.error(
            f"Account deletion stopped for user {user_id}: the Stripe cancel "
            f"failed ({type(e).__name__}, code {e.code}, request {e.request_id})"
        )
        raise PlanCancellationError() from e
    if cancelled:
        logger.info(
            f"Account deletion cancelled {cancelled} Stripe subscription(s) "
            f"of user {user_id}"
        )


async def _revoke_and_stop(user_id: str) -> None:
    """Steps before the close: an error from any of them leaves the account
    open. One connected app failing is not such an error, and
    ``stop_user_work`` never raises."""
    await revoke_connected_apps(user_id)
    await revoke_issued_access(user_id)
    await stop_user_work(user_id)
    await stop_account_emails(user_id)


async def _close(
    user_id: str, source: AccountDeletionSource
) -> AccountDeletionRequest | None:
    """Close the account; None when a concurrent call closed it first."""
    request = await close_account(user_id, source)
    # Closed either way, so this worker rejects the user from now on, no
    # background work starts for it, and its listings and creator page leave
    # the store at once (the transaction has committed).
    remember_account_closed(user_id)
    forget_refusal(user_id)
    forget_store_listings()
    if request is not None:
        logger.info(f"Account deletion closed the account of user {user_id}")
    else:
        logger.info(
            f"Account deletion for user {user_id} lost the race to a concurrent "
            f"request, which closed the account and sends the confirmation"
        )
    return request


async def _winners_erase_after(user_id: str) -> datetime:
    """The erasure date the concurrent call that closed the account set."""
    winner = await get_deletion_request(user_id)
    if winner is not None:
        return winner.eraseAfter
    # Reopened in the meantime: a fresh request would get this date.
    return datetime.now(timezone.utc) + GRACE_PERIOD


async def _send_confirmation(
    user_id: str, email: str, source: AccountDeletionSource, erase_after: datetime
) -> None:
    """Best effort: the account is already closed. One that did not go out
    stays unmarked on the request row, for the erasure routine."""
    try:
        await asyncio.wait_for(
            send_account_deleted_email(email, source, erase_after),
            timeout=CONFIRMATION_EMAIL_TIMEOUT_SECONDS,
        )
        await mark_confirmation_sent(user_id)
    except Exception as e:
        # The error type only: a delivery error can quote the address.
        logger.error(
            f"Account deletion closed the account of user {user_id} but could "
            f"not send or record the confirmation email: {type(e).__name__}"
        )
