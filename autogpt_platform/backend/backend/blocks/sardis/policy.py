from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.sardis._api import SardisClient
from backend.blocks.sardis._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    SardisCredentials,
    SardisCredentialsInput,
)
from backend.data.model import CredentialsField, SchemaField


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
        amount: float = SchemaField(
            description="Payment amount to check",
            ge=0.01,
        )
        token: str = SchemaField(
            description="Token to check (USDC, USDT, EURC, PYUSD)",
            default="USDC",
        )
        credentials: SardisCredentialsInput = CredentialsField(
            description="Sardis API credentials",
        )

    class Output(BlockSchemaOutput):
        allowed: bool = SchemaField(
            description="Whether the payment would be allowed", default=False
        )
        reason: str = SchemaField(
            description="Explanation of the policy decision", default=""
        )
        remaining_limit: float = SchemaField(
            description="Remaining spending limit after this payment", default=0
        )
        error: str = SchemaField(description="Error message if failed", default="")

    def __init__(self):
        super().__init__(
            id="f0b3c4d5-6e7f-8a9b-0c1d-2e3f4a5b6c7d",
            description="Check if a payment would pass spending policy "
            "without executing it. Useful for pre-validation.",
            categories={BlockCategory.OUTPUT},
            input_schema=SardisPolicyCheckBlock.Input,
            output_schema=SardisPolicyCheckBlock.Output,
            test_input=[
                {
                    "wallet_id": "wal_test123",
                    "destination": "0x1234567890abcdef1234567890abcdef12345678",
                    "amount": 10.0,
                    "token": "USDC",
                    "credentials": TEST_CREDENTIALS_INPUT,
                },
            ],
            test_output=[
                ("allowed", True),
                ("reason", "Payment within policy limits"),
                ("remaining_limit", 490.0),
            ],
            test_mock={
                "check_policy": lambda *args, **kwargs: {
                    "allowed": True,
                    "reason": "Payment within policy limits",
                    "remaining_limit": 490.0,
                }
            },
            test_credentials=TEST_CREDENTIALS,
        )

    @staticmethod
    async def check_policy(
        client: SardisClient,
        wallet_id: str,
        amount: float,
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
        client = SardisClient(credentials)
        result = await self.check_policy(
            client=client,
            wallet_id=input_data.wallet_id,
            amount=input_data.amount,
            destination=input_data.destination,
            token=input_data.token,
        )

        if "error" in result:
            yield "error", result["error"]
        else:
            yield "allowed", result.get("allowed", False)
            yield "reason", result.get("reason", "")
            yield "remaining_limit", float(result.get("remaining_limit", 0))
