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


# Link's customer-facing message is shown as-is on the chat card, within this.
MAX_ACTION_MESSAGE_CHARS = 300
_MAX_FIELD_CHARS = 64


class PaymentStatus(StrictModel):
    status: str
    paid: bool = False
    action_url: str = ""
    action_type: str = ""
    action_message: str = ""
    resolution: str = ""
    message: str


def payment_status(spend: SpendRequest) -> PaymentStatus:
    result = PaymentStatus(
        status=spend.status[:_MAX_FIELD_CHARS],
        # Only Link's own confirmation counts; a submitted card form does not.
        paid=spend.status == "succeeded",
        message=_MESSAGES.get(
            spend.status,
            f"Link reports status '{spend.status[:_MAX_FIELD_CHARS]}'. Treat the "
            "payment as unconfirmed and check Link.",
        ),
    )
    details = spend.status_details
    if spend.status != "requires_action" or not details or not details.requires_action:
        return result
    action = details.requires_action.next_action
    if action:
        result.action_url = link_action_url(action.action_url)
        result.action_type = action.type[:_MAX_FIELD_CHARS]
        result.action_message = action.display_message[:MAX_ACTION_MESSAGE_CHARS]
        result.resolution = action.resolution[:_MAX_FIELD_CHARS]
        if action.resolution != "auto_resume":
            result.message = (
                "This request cannot resume. Complete the action in Link; a new "
                "purchase needs fresh approval."
            )
    return result
