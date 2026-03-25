import re
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import field_validator

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.sardis._api import SardisClient, get_client
from backend.blocks.sardis._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    SardisCredentials,
    SardisCredentialsInput,
)
from backend.data.model import CredentialsField, SchemaField

_WALLET_ID_RE = re.compile(r"^wal_[a-zA-Z0-9]+$")


class SardisPolicyCheckBlock(Block):
    """Check if a payment would be allowed by spending policy before executing.

    Use this block to pre-validate payments against wallet spending policies
    without actually moving funds.
    """

    class Input(BlockSchemaInput):
        wallet_id: str = SchemaField(
            description="Sardis wallet ID (starts with wal_)",
        )
        destination: str = SchemaField(
            description="Recipient address or merchant ID",
        )
        amount: str = SchemaField(
            description=(
                "Payment amount to check as a decimal string (e.g. '25.00'). "
                "String type avoids IEEE 754 float rounding."
            ),
        )
        token: Literal["USDC", "USDT", "EURC", "PYUSD"] = SchemaField(
            description="Token to check",
            default="USDC",
        )
        credentials: SardisCredentialsInput = CredentialsField(
            description="Sardis API credentials",
        )

        @field_validator("wallet_id")
        @classmethod
        def _validate_wallet_id(cls, v: str) -> str:
            if not _WALLET_ID_RE.match(v):
                raise ValueError(
                    "wallet_id must start with 'wal_' followed by alphanumeric "
                    f"characters, got '{v}'"
                )
            return v

        @field_validator("amount")
        @classmethod
        def _validate_amount(cls, v: str) -> str:
            try:
                val = Decimal(v)
            except (InvalidOperation, TypeError):
                raise ValueError(f"amount must be a numeric string, got '{v}'")
            if not val.is_finite():
                raise ValueError(f"amount must be a finite numeric string, got '{v}'")
            if val < Decimal("0.01"):
                raise ValueError(f"amount must be >= 0.01, got '{v}'")
            return v

    class Output(BlockSchemaOutput):
        allowed: bool = SchemaField(
            description="Whether the payment would be allowed", default=False
        )
        reason: str = SchemaField(
            description="Explanation of the policy decision", default=""
        )
        remaining_limit: str = SchemaField(
            description="Remaining spending limit after this payment (decimal string)",
            default="0",
        )
        error: str = SchemaField(description="Error message if failed", default="")

    def __init__(self):
        super().__init__(
            id="37bfb8c8-4674-4362-bc27-ef6860b71f5b",
            description="Check if a payment would pass spending policy "
            "without executing it. Useful for pre-validation.",
            categories={BlockCategory.DATA},
            input_schema=SardisPolicyCheckBlock.Input,
            output_schema=SardisPolicyCheckBlock.Output,
            test_input=[
                {
                    "wallet_id": "wal_test123",
                    "destination": "0x1234567890abcdef1234567890abcdef12345678",
                    "amount": "10.00",
                    "token": "USDC",
                    "credentials": TEST_CREDENTIALS_INPUT,
                },
            ],
            test_output=[
                ("allowed", True),
                ("reason", "Payment within policy limits"),
                ("remaining_limit", "490.00"),
            ],
            test_mock={
                "check_policy": lambda *args, **kwargs: {
                    "allowed": True,
                    "reason": "Payment within policy limits",
                    "remaining_limit": "490.00",
                }
            },
            test_credentials=TEST_CREDENTIALS,
        )

    @staticmethod
    async def check_policy(
        client: SardisClient,
        wallet_id: str,
        amount: str,
        destination: str,
        token: str,
    ) -> dict:
        return await client.check_policy(
            wallet_id=wallet_id,
            amount=amount,
            destination=destination,
            token=token,
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: SardisCredentials,
        **kwargs,
    ) -> BlockOutput:
        client = get_client(credentials)
        result = await self.check_policy(
            client=client,
            wallet_id=input_data.wallet_id,
            amount=input_data.amount,
            destination=input_data.destination,
            token=input_data.token,
        )

        if "error" in result:
            yield "error", str(result["error"])
        else:
            yield "allowed", result.get("allowed", False)
            yield "reason", result.get("reason", "")
            yield "remaining_limit", str(result.get("remaining_limit", "0"))
