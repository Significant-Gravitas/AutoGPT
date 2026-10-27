"""Self-serve account deletion (SECRT-2877): close an account its owner asked
to delete.

``delete_account`` runs, in this order:

1. Load the user and check the typed email (trimmed, in any case).
2. Return at once when the account is already closed: a repeat call cancels,
   revokes and sends nothing, and records no second count.
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
6. Close the account (``close_account``): the deletion request row, the
   Better Auth ban, the end of every session and the Marketplace listings
   taken down, in one transaction. This worker rejects the user at once; the
   others within their cache TTL.
7. Send the confirmation email and record one anonymous count.

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
request row (``confirmationSentAt``); one that failed is left for the erasure
routine to send. When two calls race, the one whose ``close_account`` loses
returns without sending anything; the winner sends the email and records the
count.

Hand-off to data erasure (SECRT-2854): the AccountDeletionRequest row is the
whole hand-off. That routine (not built yet) processes every row whose
``completedAt`` is still null, so there is nothing to enqueue here. It should
first rerun ``stop_user_work``: a schedule or trigger whose removal failed here
is not retried, and neither the scheduler nor the webhook ingress checks for a
closed account. It must also delete the user's auth identities with the User
row, or the auth-identity monitor re-creates the User. For a row whose
``confirmationSentAt`` is still null it must send ``send_account_deleted_email``
(with the row's ``source``) before it erases the User row (the address goes
with it), then set the column.

Neither the typed email nor the account's email is ever logged.
"""

import asyncio
import logging

import stripe
from autogpt_libs.auth import remember_account_closed
from prisma.enums import AccountDeletionSource

from backend.api.features.account_deletion.analytics import plan_state
from backend.api.features.account_deletion.connected_apps import revoke_connected_apps
from backend.api.features.account_deletion.email_opt_out import stop_account_emails
from backend.api.features.account_deletion.issued_access import revoke_issued_access
from backend.api.features.account_deletion.scheduled_work import stop_user_work
from backend.data.account_deletion import (
    close_account,
    is_account_closed,
    mark_confirmation_sent,
)
from backend.data.credit import cancel_all_subscriptions_now, set_auto_top_up
from backend.data.model import AutoTopUpConfig
from backend.data.user import get_user_by_id
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
    AccountDeletionSource.EMAIL: "email",
}


class EmailMismatchError(Exception):
    """The typed confirmation email isn't the account's email."""


class PlanCancellationError(Exception):
    """Stripe refused or timed out on the cancel. The account is still open,
    though some of its subscriptions may already be cancelled."""


async def delete_account(
    user_id: str, confirm_email: str, source: AccountDeletionSource
) -> None:
    """Close the account, in the order the module docstring gives.

    Raises EmailMismatchError, having changed nothing, for a different email,
    and PlanCancellationError when Stripe refuses or times out on the cancel:
    the account stays open, and a retry cancels whatever is left. Any other
    error before the close also leaves the account open, so a retry is safe.
    """
    user = await get_user_by_id(user_id)
    _check_email(user.email, confirm_email)
    if await is_account_closed(user_id):
        return
    plan = plan_state(user.subscription_tier)
    await _cancel_plan(user_id)
    await set_auto_top_up(user_id, AutoTopUpConfig(threshold=0, amount=0))
    await _revoke_and_stop(user_id)
    if not await _close(user_id, source):
        return
    await _send_confirmation(user_id, user.email, source)
    capture_anonymous(
        PostHogEvent.ACCOUNT_DELETION_COMPLETED,
        {"plan_state": plan, "surface": _SURFACES[source]},
    )


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


async def _close(user_id: str, source: AccountDeletionSource) -> bool:
    """Close the account; False when a concurrent call closed it first."""
    closed = await close_account(user_id, source)
    # Closed either way, so this worker rejects the user from now on.
    remember_account_closed(user_id)
    if closed:
        logger.info(f"Account deletion closed the account of user {user_id}")
    else:
        logger.info(
            f"Account deletion for user {user_id} lost the race to a concurrent "
            f"request, which closed the account and sends the confirmation"
        )
    return closed


async def _send_confirmation(
    user_id: str, email: str, source: AccountDeletionSource
) -> None:
    """Best effort: the account is already closed. One that did not go out
    stays unmarked on the request row, for the erasure routine to send."""
    try:
        await asyncio.wait_for(
            send_account_deleted_email(email, source),
            timeout=CONFIRMATION_EMAIL_TIMEOUT_SECONDS,
        )
        await mark_confirmation_sent(user_id)
    except Exception as e:
        # The error type only: a delivery error can quote the address.
        logger.error(
            f"Account deletion closed the account of user {user_id} but could "
            f"not send or record the confirmation email: {type(e).__name__}"
        )
