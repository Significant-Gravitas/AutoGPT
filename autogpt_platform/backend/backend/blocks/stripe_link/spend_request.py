"""
Stripe Link — Spend Request blocks.

These blocks interact with the Link API (api.link.com) to create, retrieve,
and approve spend requests. A spend request provisions a one-time-use virtual
card or shared payment token from the user's Link wallet.
"""

import logging
from typing import Any, Literal

import httpx
from pydantic import model_validator

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
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
CREDENTIAL_TYPE_CARD = "card"
CREDENTIAL_TYPE_SPT = "shared_payment_token"


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
        if response.is_error:
            # Link explains itself in the body; `raise_for_status()` alone
            # reports "400 Bad Request" and throws that explanation away.
            try:
                detail = response.json().get("error", {}).get("message")
            except Exception:
                detail = None
            raise RuntimeError(
                f"Link API {response.status_code} on {method} {path}: "
                f"{detail or response.text[:200]}"
            )
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
            description=(
                "List the cards and bank accounts in the user's Link wallet. "
                "Use this first to pick a payment method ID for Create Spend "
                "Request."
            ),
            categories={BlockCategory.DATA},
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
        line_items: list[dict[str, Any]] = SchemaField(
            description=(
                "Itemised breakdown shown to the user on the approval sheet. "
                "Each item takes `name` (required) plus optional `quantity`, "
                "`unit_amount`, `description`, `sku`, `url`, `image_url` and "
                "`product_url`."
            ),
            default_factory=list,
            advanced=True,
        )
        totals: list[dict[str, Any]] = SchemaField(
            description=(
                "Total lines shown on the approval sheet. Each takes `type`, "
                "`display_text` and `amount`. `type` is one of: subtotal, tax, "
                "total, items_base_amount, items_discount, discount, "
                "fulfillment, shipping, fee, gift_wrap, tip, store_credit."
            ),
            default_factory=list,
            advanced=True,
        )
        credential_type: Literal["card", "shared_payment_token"] = SchemaField(
            description=(
                "What the spend request provisions. `card` (default) yields a "
                "one-time virtual card. `shared_payment_token` yields an SPT "
                "for merchants speaking the Machine Payments Protocol "
                "(HTTP 402), which also needs `network_id`."
            ),
            default="card",
            advanced=True,
        )
        network_id: str = SchemaField(
            description=(
                "Merchant network ID, required for `shared_payment_token`. "
                "Read it from the merchant's HTTP 402 `WWW-Authenticate: "
                "Payment` challenge."
            ),
            default="",
            advanced=True,
        )

        @model_validator(mode="after")
        def _network_id_required_for_spt(self):
            """`shared_payment_token` identifies the merchant by network ID.

            Without it we strip merchant_name/merchant_url *and* send no
            network_id, so Link receives a request with no merchant at all and
            fails obscurely. Catch it at the boundary instead.
            """
            if self.credential_type == CREDENTIAL_TYPE_SPT and not self.network_id:
                raise ValueError(
                    "network_id is required when credential_type is "
                    "'shared_payment_token' — read it from the merchant's "
                    "HTTP 402 challenge (see the Get Payment Challenge block)"
                )
            return self

        metadata: dict[str, str] = SchemaField(
            description=(
                "Arbitrary key/value data stored on the spend request. Max 50 "
                "keys; keys <= 40 chars, values <= 500 chars."
            ),
            default_factory=dict,
            advanced=True,
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
            description=(
                "Ask the user to authorize a payment, then provision a "
                "single-use credential for it. Pick the credential type by how "
                "the merchant takes payment: 'card' (default) gives a virtual "
                "card number to type into a normal checkout form; "
                "'shared_payment_token' is for merchants that answer HTTP 402 "
                "(Machine Payments Protocol), and needs the network ID from "
                "the Get Payment Challenge block. The user approves on their "
                "phone before anything is spendable."
            ),
            categories={BlockCategory.DATA},
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
        is_spt = input_data.credential_type == CREDENTIAL_TYPE_SPT
        try:
            result = await self._link_api_request(
                credentials,
                "POST",
                "/spend_requests",
                body={
                    "payment_details": input_data.payment_method_id,
                    # Rejected outright for shared_payment_token: there the
                    # merchant is identified by `network_id` from the 402
                    # challenge, not by name and URL.
                    **(
                        {}
                        if is_spt
                        else {
                            "merchant_name": input_data.merchant_name,
                            "merchant_url": input_data.merchant_url,
                        }
                    ),
                    "context": input_data.context,
                    "amount": input_data.amount,
                    "currency": input_data.currency,
                    "request_approval": input_data.request_approval,
                    "test": input_data.test_mode,
                    # Omitted when empty rather than sent as [] / {} — these
                    # drive the approval sheet's presentation, and an explicit
                    # empty is not the same as "unspecified".
                    **(
                        {"line_items": input_data.line_items}
                        if input_data.line_items
                        else {}
                    ),
                    **({"totals": input_data.totals} if input_data.totals else {}),
                    **(
                        {"metadata": input_data.metadata} if input_data.metadata else {}
                    ),
                    **(
                        {"credential_type": input_data.credential_type}
                        if input_data.credential_type != CREDENTIAL_TYPE_CARD
                        else {}
                    ),
                    **(
                        {"network_id": input_data.network_id}
                        if input_data.network_id
                        else {}
                    ),
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
            description=(
                "Fetch the unmasked virtual card number and CVC. Off by "
                "default: these are emitted as block outputs, which are "
                "persisted with the execution, so only turn it on for a graph "
                "that actually completes a card checkout."
            ),
            default=False,
        )
        include_shared_payment_token: bool = SchemaField(
            description=(
                "Include the Shared Payment Token, for spend requests created "
                "with `credential_type: shared_payment_token`."
            ),
            default=False,
        )

    class Output(BlockSchemaOutput):
        status: str = SchemaField(description="Current status of the spend request")
        card_number: str = SchemaField(
            description=(
                "Virtual card number. Single-use, capped at the approved "
                "amount, and expires at `valid_until`. Emitted only when "
                "`include_card` is on. Block outputs are persisted, so treat "
                "this as sensitive and avoid wiring it anywhere that logs."
            ),
            default="",
            secret=True,
        )
        card_cvc: str = SchemaField(
            description=(
                "Virtual card CVC. Emitted only when `include_card` is on. "
                "See the note on `card_number`: this is persisted with the "
                "execution record."
            ),
            default="",
            secret=True,
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
        shared_payment_token: str = SchemaField(
            description=(
                "One-time Shared Payment Token, when the request was created "
                "with `credential_type: shared_payment_token`. Empty otherwise. "
                "This is a bearer credential that can authorize a charge, and "
                "block outputs are persisted — treat it like the card fields."
            ),
            default="",
            secret=True,
        )
        next_action_type: str = SchemaField(
            description=(
                "Set when status is `requires_action`: what the user must "
                "resolve before approval can be requested, e.g. a 3D Secure "
                "challenge. Empty otherwise."
            ),
            default="",
        )
        next_action_message: str = SchemaField(
            description="Human-readable explanation of the required action",
            default="",
        )
        next_action_url: str = SchemaField(
            description="Where the user resolves the required action",
            default="",
        )
        auto_resumes: bool = SchemaField(
            description=(
                "True when the action resolves itself and this block can "
                "simply be polled again (3D Secure). False means a new spend "
                "request is needed once the user has acted."
            ),
            default=False,
        )
        error: str = SchemaField(
            description="Error message if the request failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="1aff59ef-e8a2-413e-9410-4ce7e4849337",
            description=(
                "Check a spend request and collect its credential once the "
                "user has approved. Returns virtual card details for a 'card' "
                "request, or the Shared Payment Token for a "
                "'shared_payment_token' one. If the status is "
                "'requires_action', the payment method needs attention first — "
                "poll again when `auto_resumes` is true, otherwise resolve the "
                "action and create a new request."
            ),
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "spend_request_id": "lsrq_test123",
                # Explicit now that it is opt-in, so the fixture still
                # exercises the card-detail path.
                "include_card": True,
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
            if input_data.include_shared_payment_token:
                include.append("shared_payment_token")
            path = f"/spend_requests/{input_data.spend_request_id}"
            if include:
                path += f"?include={','.join(include)}"

            result = await self._link_api_request(credentials, "GET", path)

            status = result["status"]
            yield "status", status

            # `requires_action` is not a failure and not an approval: the card
            # or account needs attention first. For 3D Secure the resolution is
            # `auto_resume` and the request clears itself, so the caller should
            # poll rather than give up; anything else needs a fresh request.
            if status == "requires_action":
                action = (
                    result.get("status_details", {})
                    .get("requires_action", {})
                    .get("next_action", {})
                )
                yield "next_action_type", action.get("type", "")
                yield "next_action_message", action.get("display_message", "")
                yield "next_action_url", action.get("action_url", "")
                yield "auto_resumes", action.get("resolution") == "auto_resume"

            spt = result.get("shared_payment_token") or {}
            if spt:
                yield "shared_payment_token", spt.get("id", "")

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
