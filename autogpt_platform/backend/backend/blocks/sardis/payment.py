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
    SardisCredentialsField,
    SardisCredentialsInput,
    validate_amount,
    validate_destination,
    validate_wallet_id,
)
from backend.data.model import SchemaField


class SardisPayBlock(Block):
    """Execute a policy-controlled payment from a Sardis wallet.

    Every payment is checked against configurable spending policies before
    execution. Supports USDC/USDT on Base, Polygon, Ethereum, Arbitrum, Optimism.
    """

    class Input(BlockSchemaInput):
        wallet_id: str = SchemaField(
            description="Sardis wallet ID (starts with wal_)",
        )
        destination: str = SchemaField(
            description="Recipient address, merchant ID, or wallet ID",
        )
        amount: str = SchemaField(
            description=(
                "Payment amount as a decimal string (e.g. '25.00'). "
                "String type avoids IEEE 754 float rounding."
            ),
        )
        token: Literal["USDC", "USDT", "EURC", "PYUSD"] = SchemaField(
            description="Token to use",
            default="USDC",
        )
        chain: Literal["base", "polygon", "ethereum", "arbitrum", "optimism"] = (
            SchemaField(
                description="Blockchain to use",
                default="base",
                advanced=True,
            )
        )
        purpose: str = SchemaField(
            description="Reason for payment (used in audit trail)",
            default="Payment",
            advanced=True,
        )
        credentials: SardisCredentialsInput = SardisCredentialsField()

        @field_validator("wallet_id")
        @classmethod
        def _validate_wallet_id(cls, v: str) -> str:
            return validate_wallet_id(v)

        @field_validator("destination")
        @classmethod
        def _validate_destination(cls, v: str) -> str:
            return validate_destination(v)

        @field_validator("amount")
        @classmethod
        def _validate_amount(cls, v: str) -> str:
            return validate_amount(v)

    class Output(BlockSchemaOutput):
        status: str = SchemaField(description="APPROVED, BLOCKED, or ERROR", default="")
        tx_id: str = SchemaField(description="Transaction ID if approved", default="")
        message: str = SchemaField(description="Status message", default="")
        amount: str = SchemaField(
            description="Payment amount (decimal string)", default="0"
        )

    def __init__(self):
        super().__init__(
            id="353e4e7f-f4c7-4091-badc-59170ef15500",
            description="Execute a policy-controlled payment from a Sardis wallet. "
            "Each payment is verified against spending policies before execution.",
            categories={BlockCategory.DATA},
            input_schema=SardisPayBlock.Input,
            output_schema=SardisPayBlock.Output,
            test_input=[
                {
                    "wallet_id": "wal_test123",
                    "destination": "0x1234567890abcdef1234567890abcdef12345678",
                    "amount": "10.00",
                    "token": "USDC",
                    "chain": "base",
                    "purpose": "Test payment",
                    "credentials": TEST_CREDENTIALS_INPUT,
                },
            ],
            test_output=[
                ("status", "APPROVED"),
                ("tx_id", "tx_mock123"),
                ("amount", "10.00"),
                ("message", "Payment approved"),
            ],
            test_mock={
                "send_payment": lambda *args, **kwargs: {
                    "success": True,
                    "tx_id": "tx_mock123",
                    "message": "Payment approved",
                    "amount": "10.00",
                }
            },
            test_credentials=TEST_CREDENTIALS,
            is_sensitive_action=True,
        )

    @staticmethod
    async def send_payment(
        client: SardisClient,
        wallet_id: str,
        destination: str,
        amount: str,
        token: str,
        chain: str,
        purpose: str,
        idempotency_key: str = "",
    ) -> dict:
        return await client.send_payment(
            wallet_id=wallet_id,
            to=destination,
            amount=amount,
            token=token,
            chain=chain,
            purpose=purpose,
            idempotency_key=idempotency_key,
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: SardisCredentials,
        node_exec_id: str = "",
        **kwargs,
    ) -> BlockOutput:
        client = await get_client(credentials)
        result = await self.send_payment(
            client=client,
            wallet_id=input_data.wallet_id,
            destination=input_data.destination,
            amount=input_data.amount,
            token=input_data.token,
            chain=input_data.chain,
            purpose=input_data.purpose,
            idempotency_key=node_exec_id,
        )

        # Explicit three-way status logic:
        #   1. success == True  → APPROVED
        #   2. "error" key      → ERROR (API / network / server fault)
        #   3. anything else    → BLOCKED (policy denial or unknown shape)
        if result.get("success"):
            yield "status", "APPROVED"
            yield "tx_id", result.get("tx_id", "")
            yield "amount", str(result.get("amount", input_data.amount))
            yield "message", result.get("message", "Payment approved")
        elif "error" in result:
            yield "status", "ERROR"
            yield "error", str(result["error"])
        else:
            yield "status", "BLOCKED"
            yield "message", result.get(
                "message",
                result.get("reason", "Payment blocked by policy"),
            )
