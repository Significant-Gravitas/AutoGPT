"""The checkout state machine a broker runs for one chat.

The same functions run in-process on a self-hosted deployment and behind the
broker service on hosted AutoGPT (``engine`` picks). The order of operations is
the safety argument:

1. ``create_checkout`` pins the payment fields to DOM nodes, each one a card
   input by its own declaration, and opens an approval: in Link, or in the chat
   when the customer's Link policy lets AutoGPT record it.
2. ``complete_checkout`` confirms Link approved *this* purchase, then records
   the single attempt and seals the browser (``consume_intent``) before any
   card is retrieved. A repeated or concurrent completion only reconciles.
3. The worker fills, submits once and retires the browser. Once that browser
   is gone the chat may browse again, and an attempt whose card never reached
   the page cancels its Link request.
4. Only Link's ``succeeded`` means paid, and another checkout waits until the
   attempt has a final status.
"""

import json
import time
import uuid
from pathlib import Path

from pydantic import SecretStr

from backend.util.link_checkout import broker_link
from backend.util.link_checkout.broker_protocol import (
    AuthorizedCheckout,
    CheckoutReference,
    CheckoutView,
    CreateCheckout,
    session_key,
)
from backend.util.link_checkout.cdp import prepare_browser
from backend.util.link_checkout.checkout_record import (
    archive_intent,
    consume_intent,
    current_intent,
    has_checkout,
    read_intent,
    replace_intent,
    save_intent,
    unseal,
)
from backend.util.link_checkout.config import live_payments_allowed
from backend.util.link_checkout.link import link_action_url, validate_spend
from backend.util.link_checkout.models import (
    ApprovalDetails,
    CheckoutIntent,
    SpendRequest,
    WorkerReceipt,
)
from backend.util.link_checkout.refusals import (
    ATTEMPT_UNRECONCILED,
    LIVE_PAYMENTS_DISABLED,
    CheckoutRefused,
)
from backend.util.link_checkout.runtime import (
    browser_endpoint,
    browser_operation,
    retire_payment_browser,
)
from backend.util.link_checkout.status import TERMINAL_STATUSES, payment_status

CHECKOUT_TTL_SECONDS = 600


async def create_checkout(request: CreateCheckout) -> CheckoutView:
    _require_payable(request.plan.test_mode)
    key = session_key(request)
    async with browser_operation(key) as directory:
        await _clear_finished(directory, request.access_token)
        if has_checkout(directory):
            raise CheckoutRefused(ATTEMPT_UNRECONCILED)
        intent = CheckoutIntent(
            id=uuid.uuid4().hex,
            user_id=request.user_id,
            session_id=key,
            approval_mode=request.approval_mode,
            expires_at=time.time() + CHECKOUT_TTL_SECONDS,
            plan=request.plan,
            browser=await prepare_browser(
                await browser_endpoint(directory), request.plan
            ),
        )
        save_intent(directory, intent)
        if intent.approval_mode == "in_app":
            return view(intent)
        spend = await _first_spend(directory, intent, request.access_token, None)
        return view(intent, spend)


async def complete_checkout(request: AuthorizedCheckout) -> CheckoutView:
    key = session_key(request)
    async with browser_operation(key, allow_sealed=True) as directory:
        intent = load_checkout(directory, request)
        if intent.attempted:
            return await _reconcile(directory, intent, request.access_token)
        if intent.expires_at <= time.time():
            return await _expired(directory, intent, request.access_token)
        _require_payable(intent.plan.test_mode)
        if intent.spend_request_id is not None:
            spend = await broker_link.status(intent, request.access_token)
        elif intent.approval_mode == "in_app" and request.approval is None:
            return view(intent)
        else:
            spend = await _first_spend(
                directory, intent, request.access_token, request.approval
            )
        validate_spend(intent, spend)
        if spend.status != "approved":
            return view(intent, spend)
        consume_intent(directory, intent)
        return await _attempt(directory, key, intent, spend, request.access_token)


async def reconcile(request: AuthorizedCheckout) -> CheckoutView:
    key = session_key(request)
    async with browser_operation(key, allow_sealed=True) as directory:
        intent = load_checkout(directory, request)
        if intent.spend_request_id is None:
            return view(intent)
        return await _reconcile(directory, intent, request.access_token)


async def get_checkout(request: CheckoutReference) -> CheckoutView:
    async with browser_operation(session_key(request), allow_sealed=True) as directory:
        return view(load_checkout(directory, request))


async def reset_browser(request: CheckoutReference) -> CheckoutView:
    key = session_key(request)
    async with browser_operation(key, allow_sealed=True) as directory:
        intent = load_checkout(directory, request)
        if intent.attempted and not _reconciled_final(directory, intent):
            raise CheckoutRefused(ATTEMPT_UNRECONCILED)
        if not await retire_payment_browser(key):
            raise RuntimeError("The sensitive browser has not closed")
        archive_intent(directory, intent)
        return view(intent).model_copy(
            update={
                "status": "browser_reset",
                "message": "A fresh browser is ready in this chat. The payment "
                "browser and its storage were discarded; sign in again if needed.",
            }
        )


def load_checkout(directory: Path, request: CheckoutReference) -> CheckoutIntent:
    return read_intent(
        directory,
        request.checkout_id,
        request.user_id,
        session_key(request),
        for_status=True,
    )


def view(intent: CheckoutIntent, spend: SpendRequest | None = None) -> CheckoutView:
    status, message = _pre_link_status(intent)
    result = CheckoutView(
        checkout_id=intent.id,
        spend_request_id=intent.spend_request_id,
        credentials_id=intent.plan.credentials_id,
        merchant_name=intent.plan.merchant_name,
        merchant_url=intent.plan.merchant_url(),
        amount=intent.plan.amount,
        currency=intent.plan.currency,
        test_mode=intent.plan.test_mode,
        approval_mode=intent.approval_mode,
        approval_url=intent.approval_url,
        expires_at=intent.expires_at,
        attempted=intent.attempted,
        status=status,
        message=message,
    )
    if spend:
        link = payment_status(spend)
        result.status, result.paid, result.message = (
            link.status,
            link.paid,
            link.message,
        )
        result.action_url = link.action_url
        result.action_message = link.action_message
        result.resolution = link.resolution
    return result


def _pre_link_status(intent: CheckoutIntent) -> tuple[str, str]:
    """Status and message before (or without) a fresh answer from Link."""
    if intent.attempted:
        return "outcome_unknown", "Check Link for the current payment status."
    if intent.spend_request_id is None and intent.approval_mode == "in_app":
        return (
            "awaiting_approval",
            "Waiting for the customer to approve this purchase in the chat.",
        )
    return "created", "Check Link for the approval status."


def _require_payable(test_mode: bool) -> None:
    if not test_mode and not live_payments_allowed():
        raise CheckoutRefused(LIVE_PAYMENTS_DISABLED)


async def _clear_finished(directory: Path, token: SecretStr) -> None:
    """Drop the chat's previous checkout if it is over: never attempted, so
    nothing was charged (its Link request is canceled), or attempted with a
    final status already reconciled."""
    if not has_checkout(directory):
        return
    previous = current_intent(directory)
    if not previous.attempted:
        await broker_link.cancel(previous, token)
    elif not _reconciled_final(directory, previous):
        return
    archive_intent(directory, previous)


def _reconciled_final(directory: Path, intent: CheckoutIntent) -> bool:
    path = directory / "status.json"
    status = json.loads(path.read_bytes()) if path.exists() else {}
    return (
        status.get("checkout_id") == intent.id
        and status.get("status") in TERMINAL_STATUSES
    )


async def _expired(
    directory: Path, intent: CheckoutIntent, token: SecretStr
) -> CheckoutView:
    await broker_link.cancel(intent, token)
    archive_intent(directory, intent)
    return view(intent).model_copy(
        update={
            "status": "expired",
            "message": "This checkout expired before payment; nothing was charged. "
            "Prepare a new one if the customer still wants to buy.",
        }
    )


async def _first_spend(
    directory: Path,
    intent: CheckoutIntent,
    token: SecretStr,
    approval: ApprovalDetails | None,
) -> SpendRequest:
    """Create the checkout's spend request: already approved from the
    customer's in-chat approval, or else waiting for approval in Link (also the
    retry after a lost response, which the idempotency key makes safe)."""
    spend = None
    if intent.approval_mode == "in_app" and approval is not None:
        spend = await broker_link.create_delegated(intent, token, approval)
    if spend is None:
        intent.approval_mode = "link"
        spend = await broker_link.create(intent, token)
        intent.approval_url = link_action_url(spend.approval_url)
    intent.spend_request_id = spend.id
    validate_spend(intent, spend)
    replace_intent(directory, intent)
    return spend


async def _attempt(
    directory: Path,
    key: str,
    intent: CheckoutIntent,
    spend: SpendRequest,
    token: SecretStr,
) -> CheckoutView:
    receipt = await broker_link.pay(key, intent, token)
    (directory / "receipt.json").write_text(receipt.model_dump_json())
    if receipt.browser_closed:
        # Nothing that held the card is left, so the chat may browse again;
        # another checkout still waits for this attempt's final status.
        unseal(directory)
    if receipt.status == "not_submitted":
        # The card never reached the page. Cancelling voids it and ends the
        # purchase; if Link does not confirm, the status check settles it.
        canceled = await broker_link.cancel(intent, token)
        if canceled is not None and canceled.status == "canceled":
            archive_intent(directory, intent)
            return view(intent, canceled).model_copy(
                update={
                    "receipt": receipt,
                    "attempted": True,
                    "message": "The checkout stopped before the card reached the "
                    "page, so its Link request was canceled. Nothing was charged.",
                }
            )
    return view(intent, spend).model_copy(
        update={
            "status": receipt.status,
            "receipt": receipt,
            "paid": False,
            "attempted": True,
            "message": "The payment was attempted once. Check Link status; "
            "do not submit again.",
        }
    )


async def _reconcile(
    directory: Path, intent: CheckoutIntent, token: SecretStr
) -> CheckoutView:
    spend = await broker_link.status(intent, token)
    validate_spend(intent, spend, check_deadline=False)
    (directory / "status.json").write_text(
        json.dumps({"checkout_id": intent.id, "status": spend.status})
    )
    response = view(intent, spend)
    receipt_path = directory / "receipt.json"
    if receipt_path.exists():
        response.receipt = WorkerReceipt.model_validate_json(receipt_path.read_bytes())
    return response
