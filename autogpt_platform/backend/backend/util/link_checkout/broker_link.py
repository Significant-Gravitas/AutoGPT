"""The Link calls a checkout makes, each one through the payment worker.

The worker is the one process that holds the Link token in a job and the only
one that ever holds a card; these functions start it and read back its bounded
result. They return Link's answer and leave the checkout's state to
``broker_checkout``.
"""

from pydantic import SecretStr

from backend.util.link_checkout.models import (
    ApprovalDetails,
    CheckoutIntent,
    SpendRequest,
    WorkerJob,
    WorkerReceipt,
    WorkerResult,
)
from backend.util.link_checkout.refusals import DUPLICATE_REQUEST, CheckoutRefused
from backend.util.link_checkout.runner import run_worker
from backend.util.link_checkout.runtime import retire_payment_browser


async def create(intent: CheckoutIntent, token: SecretStr) -> SpendRequest:
    """A spend request the customer approves in Link. Its idempotency key is
    the checkout's, so asking again after a lost response returns the same
    request."""
    result = await run_worker(
        WorkerJob(action="create", intent=intent, access_token=token)
    )
    return _created(result)


async def create_delegated(
    intent: CheckoutIntent, token: SecretStr, approval: ApprovalDetails
) -> SpendRequest | None:
    """A spend request Link creates already approved, from the customer's
    in-chat approval. None when Link will not take AutoGPT's approval for this
    purchase, which then needs approving in Link."""
    result = await run_worker(
        WorkerJob(
            action="create_delegated",
            intent=intent,
            access_token=token,
            approval=approval,
        )
    )
    if result.error == "link_rejected":
        return None
    return _created(result)


async def status(intent: CheckoutIntent, token: SecretStr) -> SpendRequest:
    result = await run_worker(
        WorkerJob(action="status", intent=intent, access_token=token)
    )
    if result.spend is None:
        raise RuntimeError("Link status is unavailable")
    return result.spend


async def cancel(intent: CheckoutIntent, token: SecretStr) -> SpendRequest | None:
    """Cancel the checkout's spend request, so an unused approval can neither
    be used later nor block a new request as a duplicate. Best effort: None
    when there is nothing to cancel or Link did not confirm it."""
    if intent.spend_request_id is None:
        return None
    try:
        result = await run_worker(
            WorkerJob(action="cancel", intent=intent, access_token=token)
        )
    except Exception:
        return None
    return result.spend


async def pay(key: str, intent: CheckoutIntent, token: SecretStr) -> WorkerReceipt:
    """The single payment attempt: fill, submit once, retire the browser."""
    receipt = WorkerReceipt(status="outcome_unknown")
    try:
        result = await run_worker(
            WorkerJob(action="pay", intent=intent, access_token=token)
        )
        if result.receipt:
            receipt = result.receipt
    except Exception:
        # The attempt is already recorded; an unknown outcome is reconciled
        # with Link, never retried.
        pass
    finally:
        receipt.browser_closed = await retire_payment_browser(key)
    return receipt


def _created(result: WorkerResult) -> SpendRequest:
    if result.error == "link_duplicate":
        raise CheckoutRefused(DUPLICATE_REQUEST)
    if result.spend is None:
        raise RuntimeError("The Link request's outcome is unknown")
    return result.spend
