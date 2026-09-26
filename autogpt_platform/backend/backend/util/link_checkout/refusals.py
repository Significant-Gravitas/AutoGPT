"""Checkout refusals whose reason the agent may see.

Anything else that goes wrong in a checkout is reported generically, because
its message could carry what a browser or the worker was holding. A refusal's
message is one of the fixed texts below, so it is safe to show, and a remote
broker's refusal is accepted by the controller only if it is one of them.
"""

NOT_CARD_FIELDS = (
    "The payment fields must be the page's own card inputs. Each card field "
    "needs an autocomplete of cc-number, cc-csc, and cc-exp (or cc-exp-month "
    "and cc-exp-year), and the pay button must be a button. These selectors do "
    "not qualify, so this page cannot be paid privately."
)
FIELDS_NOT_READY = (
    "The payment fields must be visible, enabled and empty, each selector must "
    "match exactly one element, and each frame URL one loaded frame."
)
FRAME_NOT_FOUND = (
    "No loaded frame has that frame URL. A frame URL must be the frame's exact "
    "address, including its query string."
)
INVALID_SELECTOR = (
    "Each selector must be plain CSS that document.querySelectorAll accepts; "
    "extensions such as :visible, :has-text() or :text-is() are not supported."
)
LIVE_PAYMENTS_DISABLED = (
    "Live Link payments are disabled by the operator; test mode is available."
)
DUPLICATE_REQUEST = (
    "Link already has a matching purchase request open from an earlier "
    "attempt. Wait about ten minutes for it to expire, or cancel it in Link, "
    "then try again."
)
ATTEMPT_UNRECONCILED = (
    "This chat's last payment attempt has no final status yet. Check it with "
    "tool:browser_link_payment_status before starting another purchase."
)
LINK_NOT_CONNECTED = (
    "Stripe Link is not connected. Run the Stripe Link List Payment Methods "
    "block with its credentials left empty: it shows the user the connect "
    "card, then lists the payment methods to choose payment_method_id from."
)
LINK_ACCOUNT_NOT_CHOSEN = (
    "Several Stripe Link accounts are connected and none was chosen in this "
    "chat. Run the Stripe Link List Payment Methods block so the user picks one."
)

MESSAGES = frozenset(
    {
        NOT_CARD_FIELDS,
        FIELDS_NOT_READY,
        FRAME_NOT_FOUND,
        INVALID_SELECTOR,
        LIVE_PAYMENTS_DISABLED,
        DUPLICATE_REQUEST,
        ATTEMPT_UNRECONCILED,
        LINK_NOT_CONNECTED,
        LINK_ACCOUNT_NOT_CHOSEN,
    }
)


class CheckoutRefused(ValueError):
    """The checkout was refused for a reason in ``MESSAGES``; nothing was paid."""

    def __init__(self, message: str):
        if message not in MESSAGES:
            raise ValueError("Unknown checkout refusal")
        super().__init__(message)
