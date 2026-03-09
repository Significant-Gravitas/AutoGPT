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
        amount: float = SchemaField(
            description="Payment amount in token units",
            ge=0.01,
        )
        token: str = SchemaField(
            description="Token to use (USDC, USDT, EURC, PYUSD)",
            default="USDC",
        )
        chain: str = SchemaField(
            description="Blockchain to use (base, polygon, ethereum, arbitrum, optimism)",
            default="base",
            advanced=True,
        )
        purpose: str = SchemaField(
            description="Reason for payment (used in audit trail)",
            default="Payment",
            advanced=True,
        )
        credentials: SardisCredentialsInput = CredentialsField(
            description="Sardis API credentials",
        )

    class Output(BlockSchemaOutput):
        status: str = SchemaField(description="APPROVED, BLOCKED, or ERROR")
        tx_id: str = SchemaField(
            description="Transaction ID if approved", default=""
        )
        message: str = SchemaField(description="Status message", default="")
        amount: float = SchemaField(description="Payment amount", default=0)
        error: str = SchemaField(description="Error message if failed", default="")

    def __init__(self):
        super().__init__(
            id="d8f1a2b3-4c5d-6e7f-8a9b-0c1d2e3f4a5b",
            description="Execute a policy-controlled payment from a Sardis wallet. "
            "Each payment is verified against spending policies before execution.",
            categories={BlockCategory.OUTPUT},
            input_schema=SardisPayBlock.Input,
            output_schema=SardisPayBlock.Output,
            test_input=[
                {
                    "wallet_id": "wal_test123",
                    "destination": "0x1234567890abcdef1234567890abcdef12345678",
                    "amount": 10.0,
                    "token": "USDC",
                    "chain": "base",
                    "purpose": "Test payment",
                    "credentials": TEST_CREDENTIALS_INPUT,
                },
            ],
            test_output=[
                ("status", "APPROVED"),
                ("tx_id", "tx_mock123"),
                ("amount", 10.0),
            ],
            test_mock={
                "send_payment": lambda *args, **kwargs: {
                    "success": True,
                    "tx_id": "tx_mock123",
                    "message": "Payment approved",
                    "amount": 10.0,
                }
            },
            test_credentials=TEST_CREDENTIALS,
        )

    @staticmethod
    async def send_payment(
        client: SardisClient,
        wallet_id: str,
        destination: str,
        amount: float,
        token: str,
        chain: str,
        purpose: str,
    ) -> dict:
        return await client.send_payment(
            wallet_id=wallet_id,
            to=destination,
            amount=amount,
            token=token,
            chain=chain,
            purpose=purpose,
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: SardisCredentials,
        **kwargs,
    ) -> BlockOutput:
        client = SardisClient(credentials)
        result = await self.send_payment(
            client=client,
            wallet_id=input_data.wallet_id,
            destination=input_data.destination,
            amount=input_data.amount,
            token=input_data.token,
            chain=input_data.chain,
            purpose=input_data.purpose,
        )

        if result.get("success"):
            yield "status", "APPROVED"
            yield "tx_id", result.get("tx_id", "")
            yield "amount", input_data.amount
            yield "message", result.get("message", "Payment approved")
        elif result.get("error"):
            yield "status", "ERROR"
            yield "error", result.get("error", "Unknown error")
        else:
            yield "status", "BLOCKED"
            yield "message", result.get("message", "Payment blocked by policy")
