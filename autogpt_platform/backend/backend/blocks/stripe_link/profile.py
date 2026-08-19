"""
Stripe Link — wallet profile blocks.

Read-only lookups an agent needs to actually complete a purchase: who the
user is, and where to ship. Both are separate from the spend-request flow.
"""

from typing import Any

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
    link_api_request,
)
from backend.data.model import SchemaField


class StripeLinkGetUserInfoBlock(Block):
    """Read the Link account holder's name and contact details."""

    # Exposed as a class attribute so `test_mock` can patch it; the harness
    # only replaces names it can find on the block instance.
    _link_api_request = staticmethod(link_api_request)

    class Input(BlockSchemaInput):
        credentials: StripeLinkCredentialsInput = StripeLinkCredentialsField()

    class Output(BlockSchemaOutput):
        name: str = SchemaField(description="Full name on the Link account")
        first_name: str = SchemaField(description="Given name", default="")
        last_name: str = SchemaField(description="Family name", default="")
        email: str = SchemaField(description="Email on the Link account", default="")
        phone: str = SchemaField(
            description="Phone number in E.164 format, e.g. +15551234567", default=""
        )
        error: str = SchemaField(
            description="Error message if the request failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="780daefe-be88-457d-af4b-c8c931daaad0",
            description=(
                "Get the Link account holder's name, email and phone. Use it "
                "to fill in a checkout that asks who the buyer is. Pairs with "
                "Get Shipping Address for anything physical."
            ),
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("name", "Ada Lovelace"),
                ("first_name", "Ada"),
                ("last_name", "Lovelace"),
                ("email", "ada@example.com"),
                ("phone", "+15551234567"),
            ],
            test_mock={
                "_link_api_request": lambda *args, **kwargs: {
                    "name": "Ada Lovelace",
                    "first_name": "Ada",
                    "last_name": "Lovelace",
                    "email": "ada@example.com",
                    "phone": "+15551234567",
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
            info = await self._link_api_request(credentials, "GET", "/userinfo")
            yield "name", info.get("name", "")
            yield "first_name", info.get("first_name", "")
            yield "last_name", info.get("last_name", "")
            yield "email", info.get("email", "")
            yield "phone", info.get("phone", "")
        except Exception as e:
            yield "error", str(e)


class StripeLinkGetShippingAddressBlock(Block):
    """Read the shipping addresses saved on the Link account."""

    _link_api_request = staticmethod(link_api_request)

    class Input(BlockSchemaInput):
        credentials: StripeLinkCredentialsInput = StripeLinkCredentialsField()

    class Output(BlockSchemaOutput):
        addresses: list[dict[str, Any]] = SchemaField(
            description="Every shipping address on the account, each with an "
            "`id`, `is_default` and an `address` object"
        )
        default_address: dict[str, Any] = SchemaField(
            description="The address object marked default, or the first one "
            "if none is. Empty when the account has no addresses."
        )
        error: str = SchemaField(
            description="Error message if the request failed", default=""
        )

    def __init__(self):
        example = {
            "id": "csmrsa_test",
            "is_default": True,
            "address": {
                "line_1": "1 Infinite Loop",
                "locality": "Cupertino",
                "administrative_area": "CA",
                "postal_code": "95014",
                "country_code": "US",
            },
        }
        super().__init__(
            id="4ace99c4-5b0d-4f2d-a371-397c65acfcd2",
            description=(
                "Get the delivery addresses saved on the user's Link wallet, "
                "with the default one resolved for you. Use it for any "
                "purchase that ships something."
            ),
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("addresses", [example]),
                ("default_address", example["address"]),
            ],
            test_mock={
                "_link_api_request": lambda *args, **kwargs: {
                    "shipping_addresses": [example]
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
            response = await self._link_api_request(
                credentials, "GET", "/shipping_addresses"
            )
            addresses = response.get("shipping_addresses", [])
            yield "addresses", addresses

            # Prefer the one the user marked default; fall back to the first so
            # a single-address account still yields something usable.
            default = next(
                (a for a in addresses if a.get("is_default")),
                addresses[0] if addresses else None,
            )
            yield "default_address", (default or {}).get("address", {})
        except Exception as e:
            yield "error", str(e)
