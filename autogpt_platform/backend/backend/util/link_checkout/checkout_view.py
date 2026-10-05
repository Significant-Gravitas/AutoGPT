"""What a checkout looks like to the agent and the chat card: the engine's own
state, overlaid with Link's latest status when there is one."""

from backend.util.link_checkout.broker_protocol import CheckoutView
from backend.util.link_checkout.models import CheckoutIntent, SpendRequest
from backend.util.link_checkout.status import payment_status


def view(intent: CheckoutIntent, spend: SpendRequest | None = None) -> CheckoutView:
    status, message = _pre_link_status(intent)
    result = CheckoutView(
        checkout_id=intent.id,
        spend_request_id=intent.spend_request_id,
        credentials_id=intent.plan.credentials_id,
        payment_method_id=intent.plan.payment_method_id,
        merchant_name=intent.plan.merchant_name,
        merchant_url=intent.plan.merchant_url(),
        amount=intent.plan.amount,
        currency=intent.plan.currency,
        test_mode=intent.plan.test_mode,
        approval_mode=intent.approval_mode,
        approval_url=intent.approval_url,
        expires_at=intent.expires_at,
        attempted=intent.attempted,
        revision=intent.revision,
        status=status,
        message=message,
    )
    if spend:
        link = payment_status(
            spend, attempted=intent.attempted, test_mode=intent.plan.test_mode
        )
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
