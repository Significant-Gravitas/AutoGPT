"""
Stripe Link — Spend Request blocks.

These blocks interact with the Link API (api.link.com) to create, retrieve,
and approve spend requests. A spend request provisions a one-time-use virtual
card or shared payment token from the user's Link wallet.
"""

import logging
from typing import Any

from backend.blocks._base import Block, BlockOutput, BlockSchemaInput, BlockSchemaOutput
from backend.blocks.stripe_link._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    StripeLinkCredentials,
    StripeLinkCredentialsField,
    StripeLinkCredentialsInput,
)
from backend.data.model import SchemaField

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
LINK_API_BASE = "https://api.link.com"


async def link_api_request(
    credentials: StripeLinkCredentials,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Make an authenticated request to the Link API.

    Uses the access_token from OAuth2Credentials as a Bearer token.

    Refresh is deliberately not handled here: `IntegrationCredentialsManager`
    already refreshes on acquire (`_refresh_locked`), under a per-credential
    lock, and persists the rotated tokens. Refreshing inside the block would
    bypass both and let concurrent nodes stampede the token endpoint.
    """
    import httpx

    headers = {
        "Authorization": f"Bearer {credentials.access_token.get_secret_value()}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient() as client:
        response = await client.request(
            method=method,
            url=f"{LINK_API_BASE}{path}",
            headers=headers,
            json=body,
        )
        response.raise_for_status()
        return response.json()


# ---------------------------------------------------------------------------
# Block: List Payment Methods
# ---------------------------------------------------------------------------
class StripeLinkListPaymentMethodsBlock(Block):
    """List payment methods (cards and bank accounts) from the user's Link wallet."""

    # Exposed as a class attribute so `test_mock` can patch it; the harness
    # only replaces names it can find on the block instance.
    _link_api_request = staticmethod(link_api_request)

    class Input(BlockSchemaInput):
        credentials: StripeLinkCredentialsInput = StripeLinkCredentialsField()

    class Output(BlockSchemaOutput):
        payment_methods: list[dict[str, Any]] = SchemaField(
            description="List of payment methods in the Link wallet"
        )
        error: str = SchemaField(
            description="Error message if the request failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="6eacc954-2218-4dc7-a485-5bf21549ecbe",
            description="List payment methods from a Stripe Link wallet",
            categories=set(),
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "payment_methods",
                    [
                        {
                            "id": "csmrpd_test",
                            "type": "CARD",
                            "name": "Test Debit Card",
                            "is_default": False,
                            "card_details": {
                                "brand": "visa",
                                "last4": "4242",
                                "exp_month": 12,
                                "exp_year": 2030,
                            },
                        }
                    ],
                )
            ],
            test_mock={
                "_link_api_request": lambda *args, **kwargs: {
                    "payment_details": [
                        {
                            "id": "csmrpd_test",
                            "type": "CARD",
                            "name": "Test Debit Card",
                            "is_default": False,
                            "card_details": {
                                "brand": "visa",
                                "last4": "4242",
                                "exp_month": 12,
                                "exp_year": 2030,
                            },
                        }
                    ]
                }
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: StripeLinkCredentials,
        **kwargs: Any,
    ) -> BlockOutput:
        try:
            # `/payment-details`, not `/payment_methods` — the latter 404s. The
            # list is nested under `payment_details`, matching @stripe/link-cli's
            # SDK (packages/sdk/src/resources/payment-methods.ts).
            response = await self._link_api_request(
                credentials, "GET", "/payment-details"
            )
            yield "payment_methods", response.get("payment_details", [])
        except Exception as e:
            yield "error", str(e)


# ---------------------------------------------------------------------------
# Block: Create Spend Request
# ---------------------------------------------------------------------------
class StripeLinkCreateSpendRequestBlock(Block):
    """
    Create a spend request to get a one-time-use payment credential.

    The user must approve the request via the Link app before card details
    are available. Use StripeLinkRetrieveSpendRequestBlock to check status
    and get the credential once approved.
    """

    # Exposed as a class attribute so `test_mock` can patch it; the harness
    # only replaces names it can find on the block instance.
    _link_api_request = staticmethod(link_api_request)

    class Input(BlockSchemaInput):
        credentials: StripeLinkCredentialsInput = StripeLinkCredentialsField()
        payment_method_id: str = SchemaField(
            description="ID of the payment method to use (from list payment methods)"
        )
        merchant_name: str = SchemaField(
            description="Name of the merchant for this purchase"
        )
        merchant_url: str = SchemaField(description="URL of the merchant website")
        context: str = SchemaField(
            description=(
                "Description of the purchase context (min 100 characters). "
                "Shown to the user when they approve the request."
            ),
            # Enforced, not just documented: this is the text the user reads
            # when deciding whether to approve a charge.
            min_length=100,
        )
        amount: int = SchemaField(
            description="Amount in cents (max 50000)", ge=1, le=50000
        )
        currency: str = SchemaField(
            description="3-letter ISO currency code", default="usd"
        )
        request_approval: bool = SchemaField(
            description=(
                "If true, immediately sends a push notification to the user "
                "for approval. Otherwise, call request-approval separately."
            ),
            default=True,
        )
        test_mode: bool = SchemaField(
            description="Use test mode (fake card 4242424242424242)",
            default=False,
        )

    class Output(BlockSchemaOutput):
        spend_request_id: str = SchemaField(
            description="ID of the created spend request"
        )
        status: str = SchemaField(
            description="Status: created, pending_approval, approved, denied, etc."
        )
        approval_url: str = SchemaField(
            description="URL the user can visit to approve (if not using push)",
            default="",
        )
        error: str = SchemaField(
            description="Error message if the request failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="932c3c12-1e80-4392-8fb3-37824eb8a427",
            description="Create a Stripe Link spend request for a one-time payment credential",
            categories=set(),
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "payment_method_id": "csmrpd_test",
                "merchant_name": "Test Store",
                "merchant_url": "https://example.com",
                "context": "x" * 100,
                "amount": 1000,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("spend_request_id", "lsrq_test123"),
                ("status", "pending_approval"),
            ],
            test_mock={
                "_link_api_request": lambda *args, **kwargs: {
                    "id": "lsrq_test123",
                    "status": "pending_approval",
                    "approval_url": "",
                }
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: StripeLinkCredentials,
        **kwargs: Any,
    ) -> BlockOutput:
        try:
            result = await self._link_api_request(
                credentials,
                "POST",
                "/spend_requests",
                body={
                    "payment_details": input_data.payment_method_id,
                    "merchant_name": input_data.merchant_name,
                    "merchant_url": input_data.merchant_url,
                    "context": input_data.context,
                    "amount": input_data.amount,
                    "currency": input_data.currency,
                    "request_approval": input_data.request_approval,
                    "test": input_data.test_mode,
                },
            )
            # Note: do NOT also call POST /spend_requests/{id}/request_approval
            # here. `request_approval` in the body already moves the request to
            # pending_approval, and the dedicated endpoint then 409s. That
            # endpoint is for requesting approval on a request created without
            # it — verified against the live API.
            yield "spend_request_id", result["id"]
            yield "status", result["status"]
            if result.get("approval_url"):
                yield "approval_url", result["approval_url"]
        except Exception as e:
            yield "error", str(e)


# ---------------------------------------------------------------------------
# Block: Retrieve Spend Request
# ---------------------------------------------------------------------------
class StripeLinkRetrieveSpendRequestBlock(Block):
    """
    Retrieve a spend request and its credentials (once approved).

    After the user approves a spend request, this block returns the
    virtual card details (number, CVC, expiry, billing address) that
    can be used for a one-time purchase.
    """

    # Exposed as a class attribute so `test_mock` can patch it; the harness
    # only replaces names it can find on the block instance.
    _link_api_request = staticmethod(link_api_request)

    class Input(BlockSchemaInput):
        credentials: StripeLinkCredentialsInput = StripeLinkCredentialsField()
        spend_request_id: str = SchemaField(
            description="ID of the spend request to retrieve (e.g., lsrq_...)"
        )
        include_card: bool = SchemaField(
            description="Include unmasked card details in the response",
            default=True,
        )

    class Output(BlockSchemaOutput):
        status: str = SchemaField(description="Current status of the spend request")
        card_number: str = SchemaField(
            description="Virtual card number (only if approved and include_card=True)",
            default="",
        )
        card_cvc: str = SchemaField(
            description="Virtual card CVC",
            default="",
        )
        card_exp_month: int = SchemaField(
            description="Card expiry month",
            default=0,
        )
        card_exp_year: int = SchemaField(
            description="Card expiry year",
            default=0,
        )
        card_brand: str = SchemaField(
            description="Card brand (visa, mastercard, etc.)",
            default="",
        )
        valid_until: str = SchemaField(
            description="ISO timestamp when the virtual card expires",
            default="",
        )
        error: str = SchemaField(
            description="Error message if the request failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="1aff59ef-e8a2-413e-9410-4ce7e4849337",
            description="Retrieve a Stripe Link spend request and card credentials",
            categories=set(),
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "spend_request_id": "lsrq_test123",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("status", "approved"),
                ("card_number", "4242424242424242"),
                ("card_cvc", "123"),
                ("card_exp_month", 12),
                ("card_exp_year", 2030),
                ("card_brand", "visa"),
                ("valid_until", "2025-12-31T23:59:59Z"),
            ],
            test_mock={
                "_link_api_request": lambda *args, **kwargs: {
                    "status": "approved",
                    "card": {
                        "number": "4242424242424242",
                        "cvc": "123",
                        "exp_month": 12,
                        "exp_year": 2030,
                        "brand": "visa",
                        "valid_until": "2025-12-31T23:59:59Z",
                    },
                }
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: StripeLinkCredentials,
        **kwargs: Any,
    ) -> BlockOutput:
        try:
            include = ["card"] if input_data.include_card else []
            path = f"/spend_requests/{input_data.spend_request_id}"
            if include:
                path += f"?include={','.join(include)}"

            result = await self._link_api_request(credentials, "GET", path)

            yield "status", result["status"]

            card = result.get("card")
            if card:
                yield "card_number", card.get("number", "")
                yield "card_cvc", card.get("cvc", "")
                yield "card_exp_month", card.get("exp_month", 0)
                yield "card_exp_year", card.get("exp_year", 0)
                yield "card_brand", card.get("brand", "")
                yield "valid_until", card.get("valid_until", "")
        except Exception as e:
            yield "error", str(e)
