import re

from backend.util.link_checkout.link import link_action_url
from backend.util.link_checkout.models import SpendRequest, StrictModel

# Statuses after which the spend request can never pay again.
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "denied", "expired", "canceled"})

_MESSAGES = {
    "created": "Link is preparing the approval request.",
    "pending_approval": "Review this purchase in Link.",
    "approved": "Link approved this request. Approval alone does not confirm payment.",
    "submitted": "Link reports the payment is processing. Check again shortly.",
    "succeeded": "Link confirms the payment completed.",
    "failed": "Link reports that this payment failed. Review the request before "
    "starting another purchase.",
    "denied": "The purchase was declined in Link.",
    "expired": "This Link request expired.",
    "canceled": "This Link request was canceled.",
    "requires_action": "Complete the required action in Link, then check this "
    "request again.",
}

# Link's `requires_action` guidance. What happens next follows the resolution
# (Link: "branch on resolution, not on type"); the type only words the step.
# Only `auto_resume` keeps the request alive; the others end it, and it then
# expires on its own.
# https://docs.stripe.com/agentic-commerce/link-agent-wallet/use-link-wallet-pay-online#next-actions
AUTO_RESUME = "auto_resume"
_BY_RESOLUTION = {
    AUTO_RESUME: "The customer must finish a step in Link; this request resumes "
    "once they do. Give them the link, then check the status again.",
    "create_new_spend_request": "This request can no longer pay. Fix what Link "
    "reported, then prepare a new checkout.",
    "create_new_spend_request_after_completion": "This request can no longer "
    "pay. Give the customer the link to finish what Link needs, then prepare a "
    "new checkout.",
}
_BY_TYPE = {
    "three_d_secure": "The card issuer wants the customer to verify this "
    "purchase. Give them the verification link; the request resumes once they "
    "finish, so check the status again afterwards.",
    "three_d_secure_retry": "The customer did not finish verifying this "
    "purchase. If they still want it, prepare a new checkout and ask them to "
    "complete the verification when it appears.",
    "select_payment_method": "The payment method was declined. Ask the customer "
    "to pick another Link payment method, then prepare a new checkout with it.",
    "update_payment_method": "The payment method needs updating in Link. Ask the "
    "customer to update it, then prepare a new checkout.",
    "re_authorize": "The charge was more than the customer approved. Prepare a "
    "new checkout for the correct total.",
    "add_payment_method": "The customer has no Link payment method that can pay. "
    "Give them the link to add one, then prepare a new checkout.",
    "ssn_verification": "Link needs the customer to verify their identity. Give "
    "them the link; once they finish, prepare a new checkout.",
    "identity_verification": "Link needs the customer to verify their identity. "
    "Give them the link; once they finish, prepare a new checkout.",
    "contact_support": "Link needs the customer to contact Link support. Give "
    "them the link; once it is resolved, prepare a new checkout.",
}
# The resolution each type comes with today. A type's wording is used only
# when Link pairs it with that resolution, so the words never promise a
# request will resume when Link says it ends, or the reverse.
_TYPE_RESOLUTION = {
    "three_d_secure": AUTO_RESUME,
    "three_d_secure_retry": "create_new_spend_request",
    "select_payment_method": "create_new_spend_request",
    "update_payment_method": "create_new_spend_request",
    "re_authorize": "create_new_spend_request",
    "add_payment_method": "create_new_spend_request_after_completion",
    "ssn_verification": "create_new_spend_request_after_completion",
    "identity_verification": "create_new_spend_request_after_completion",
    "contact_support": "create_new_spend_request_after_completion",
}

TEST_PAYMENT_SUBMITTED = (
    "Test payment: the card was submitted once. Link never charges test cards, "
    "so this request stays 'approved' and never reaches 'succeeded'. Only the "
    "store's own confirmation shows whether it accepted the order."
)

# Link's customer-facing message is shown as-is on the chat card, within this.
MAX_ACTION_MESSAGE_CHARS = 300
_MAX_FIELD_CHARS = 64
_CODE = re.compile(r"^[a-z0-9_]{1,64}$")


class PaymentStatus(StrictModel):
    status: str
    paid: bool = False
    # Nothing more will happen to this request: a terminal status, or an
    # action that ends it.
    final: bool = False
    action_url: str = ""
    action_message: str = ""
    resolution: str = ""
    message: str


def payment_status(
    spend: SpendRequest, *, attempted: bool = False, test_mode: bool = False
) -> PaymentStatus:
    status = spend.status[:_MAX_FIELD_CHARS]
    result = PaymentStatus(
        status=status,
        # Only Link's own confirmation counts; a submitted card form does not.
        paid=spend.status == "succeeded",
        final=spend.status in TERMINAL_STATUSES,
        message=_MESSAGES.get(
            spend.status,
            f"Link reports status '{status}'. Treat the payment as unconfirmed "
            "and check Link.",
        ),
    )
    if spend.status == "failed":
        result.message = _failure_message(spend)
    if attempted and test_mode and spend.status == "approved":
        # A test card is never charged, so this is as far as it goes.
        result.final = True
        result.message = TEST_PAYMENT_SUBMITTED
    details = spend.status_details
    if spend.status != "requires_action" or not details or not details.requires_action:
        return result
    action = details.requires_action.next_action
    if action:
        resolution = action.resolution[:_MAX_FIELD_CHARS]
        result.action_url = link_action_url(action.action_url)
        result.action_message = action.display_message[:MAX_ACTION_MESSAGE_CHARS]
        result.resolution = resolution
        result.final = resolution != AUTO_RESUME
        result.message = (
            _BY_TYPE[action.type]
            if _TYPE_RESOLUTION.get(action.type) == resolution
            else _BY_RESOLUTION.get(
                resolution, _BY_RESOLUTION["create_new_spend_request"]
            )
        )
    return result


def _failure_message(spend: SpendRequest) -> str:
    details = spend.payment_status_details
    reasons = [
        f"{label} {value}"
        for label, value in (
            ("decline code", details and details.decline_code),
            ("code", details and details.code),
        )
        if value and _CODE.match(value)
    ]
    reason = f" ({', '.join(reasons)})" if reasons else ""
    return (
        f"Link reports that this payment failed{reason}. Nothing more will be "
        "charged on this request; prepare a new checkout if the customer still "
        "wants the purchase."
    )
