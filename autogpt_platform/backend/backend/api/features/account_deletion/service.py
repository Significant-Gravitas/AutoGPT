"""Self-serve account deletion (SECRT-2877): close an account its owner asked
to delete.

``delete_account`` runs, in this order:

1. Load the user and check the typed email (trimmed, in any case).
2. Return at once when the account is already closed: a repeat call cancels,
   revokes and sends nothing, and records no second count.
3. Read the plan for the completed count, before Stripe is touched: the
   cancel's webhooks move the tier, and can land while this still runs.
4. Cancel every Stripe subscription now, with no invoice. A Stripe error stops
   everything here, and the person is told nothing was deleted.
5. Disconnect the apps they connected, revoke the API keys and OAuth grants
   AutoGPT issued for them, stop their schedules, triggers, AutoPilot turns
   and runs, and switch off their product and marketing email.
6. Close the account (``close_account``): the deletion request row, the
   Better Auth ban and the end of every session, in one transaction. This
   worker rejects the user at once; the others within their cache TTL.
7. Send the confirmation email and record one anonymous count.

Every step before the close must succeed or raise, so the account never
closes with a plan still billing or an app still connected. Nothing is closed
until step 6, so the person can simply try again, and each step is safe to
repeat:

- the Stripe cancel lists only subscriptions Stripe can still cancel, so a
  retry cancels just what the first attempt missed;
- ``revoke_connected_apps`` works from the credentials still stored: the ones
  already deleted are gone, and the managed-credential cleanup logs, rather
  than raises, a resource that is already gone;
- ``revoke_issued_access`` revokes only keys and grants not revoked yet;
- ``stop_user_work`` lists what is still scheduled or running, and never
  raises;
- ``stop_account_emails`` writes the same settings again, and an earlier
  marketing opt-out keeps its date and source.

The steps after the close are best effort: the account is closed, so the call
succeeds even when the email cannot be sent. When two calls race, the one whose
``close_account`` loses returns without sending anything; the winner sends the
email and records the count.

Hand-off to data erasure (SECRT-2854): the AccountDeletionRequest row is the
whole hand-off. That routine (not built yet) processes every row whose
``completedAt`` is still null, so there is nothing to enqueue here.

Neither the typed email nor the account's email is ever logged.
"""

import logging

import stripe
from autogpt_libs.auth import remember_account_closed
from prisma.enums import AccountDeletionSource

from backend.api.features.account_deletion.analytics import plan_state
from backend.api.features.account_deletion.connected_apps import revoke_connected_apps
from backend.api.features.account_deletion.email_opt_out import stop_account_emails
from backend.api.features.account_deletion.issued_access import revoke_issued_access
from backend.api.features.account_deletion.scheduled_work import stop_user_work
from backend.data.account_deletion import close_account, is_account_closed
from backend.data.credit import cancel_all_subscriptions_now
from backend.data.user import get_user_by_id
from backend.notifications.auth import send_account_deleted_email
from backend.util.posthog_client import capture_anonymous
from backend.util.posthog_events import PostHogEvent

logger = logging.getLogger(__name__)

# Where the flow was opened, as the completed count reports it. Not "source":
# that is the base property every event carries.
_SURFACES = {
    AccountDeletionSource.SETTINGS: "settings",
    AccountDeletionSource.PAYWALL: "paywall",
}


class EmailMismatchError(Exception):
    """The typed confirmation email isn't the account's email."""


class PlanCancellationError(Exception):
    """Stripe refused to cancel the plan; nothing else was changed."""


async def delete_account(
    user_id: str, confirm_email: str, source: AccountDeletionSource
) -> None:
    """Close the account, in the order the module docstring gives.

    Raises EmailMismatchError for a different email and PlanCancellationError
    when Stripe refuses the cancel; in both cases nothing has changed. Any
    other error before the close leaves the account open, so a retry is safe.
    """
    user = await get_user_by_id(user_id)
    _check_email(user.email, confirm_email)
    if await is_account_closed(user_id):
        return
    plan = plan_state(user.subscription_tier)
    await _cancel_plan(user_id)
    await _revoke_and_stop(user_id)
    if not await _close(user_id, source):
        return
    await _send_confirmation(user_id, user.email)
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
    """Steps that must succeed or raise before the account closes."""
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


async def _send_confirmation(user_id: str, email: str) -> None:
    """Best effort: the account is already closed."""
    try:
        await send_account_deleted_email(email)
    except Exception as e:
        # The error type only: a delivery error can quote the address.
        logger.error(
            f"Account deletion closed the account of user {user_id} but could "
            f"not send the confirmation email: {type(e).__name__}"
        )
