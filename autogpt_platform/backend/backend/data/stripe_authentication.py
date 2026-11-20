"""Whether a refused Stripe charge only needs the customer's bank to confirm it."""

import stripe

# The card is fine but the bank wants 3DS. Stripe documents
# ``authentication_required`` (also a ``card_declined`` decline code) and
# ``invoice_payment_intent_requires_action`` for an invoice payment that needs
# action; ``subscription_payment_intent_requires_action`` is kept because it has
# been reported for Subscription.modify under ``error_if_incomplete``.
_SCA_CARD_ERROR_CODES = frozenset(
    {
        "authentication_required",
        "invoice_payment_intent_requires_action",
        "subscription_payment_intent_requires_action",
    }
)


def requires_authentication(error: stripe.CardError) -> bool:
    decline_code = error.error.decline_code if error.error else None
    return (
        error.code in _SCA_CARD_ERROR_CODES or decline_code == "authentication_required"
    )
