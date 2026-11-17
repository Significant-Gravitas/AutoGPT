"""The checkout tools' parameters, as the model sees them."""

from typing import Any

# Written out rather than generated: the model sees plain selector strings,
# and ``CheckoutPlan`` enforces the patterns and bounds when the call arrives.
REQUEST_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "credentials_id": {
            "type": "string",
            "description": "Stripe Link credential to pay with. Leave it out to "
            "use the account connected in this chat.",
        },
        "payment_method_id": {
            "type": "string",
            "description": "Link payment method to pay with (csmrpd_...).",
        },
        "merchant_name": {"type": "string", "description": "Merchant the user pays."},
        "checkout_url": {
            "type": "string",
            "description": "Exact HTTPS URL of the open checkout tab.",
        },
        "amount": {
            "type": "integer",
            "description": "Final total in the currency's smallest unit (1250 = 12.50).",
        },
        "currency": {"type": "string", "description": "Lowercase ISO code; usd."},
        "context": {
            "type": "string",
            "description": "What is being bought and why, shown to the user "
            "(at least 100 characters).",
        },
        "execution": {
            "type": "string",
            "enum": ["card", "link_pay_token"],
            "description": "card (default), or link_pay_token on a Stripe "
            "checkout with an 'I am an AI agent' option: name only submit.",
        },
        "number": {"type": "string", "description": "Selector of the card number."},
        "cvc": {"type": "string", "description": "Selector of the CVC."},
        "expiry": {
            "type": "string",
            "description": "Selector of a combined MM/YY field; else give "
            "exp_month and exp_year.",
        },
        "exp_month": {"type": "string", "description": "Selector of the month."},
        "exp_year": {"type": "string", "description": "Selector of the year."},
        "submit": {"type": "string", "description": "Selector of the pay button."},
        "frame_urls": {
            "type": "object",
            "description": "For fields inside an iframe: field name (number, cvc, "
            "expiry, exp_month, exp_year, submit) to the frame's exact URL.",
            "additionalProperties": {"type": "string"},
        },
        "test_mode": {
            "type": "boolean",
            "description": "Test payment with no charge. Default true.",
        },
    },
    "required": [
        "payment_method_id",
        "merchant_name",
        "checkout_url",
        "amount",
        "context",
        "submit",
    ],
}

RAISE_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "checkout_id": {
            "type": "string",
            "description": "checkout_id from tool:browser_request_link_payment.",
        },
        "amount": {
            "type": "integer",
            "description": "New total in the smallest unit; higher than before.",
        },
        "reason": {"type": "string", "description": "Why it rose, shown to the user."},
    },
    "required": ["checkout_id", "amount", "reason"],
}

CHECKOUT_ID_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "checkout_id": {
            "type": "string",
            "description": "checkout_id from tool:browser_request_link_payment.",
        }
    },
    "required": ["checkout_id"],
}
