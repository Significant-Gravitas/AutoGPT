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


class SardisBalanceBlock(Block):
    """Check the balance and spending limits of a Sardis wallet."""

    class Input(BlockSchemaInput):
        wallet_id: str = SchemaField(
            description="Sardis wallet ID (starts with wal_)",
        )
        token: str = SchemaField(
            description="Token to check (USDC, USDT, EURC, PYUSD)",
            default="USDC",
        )
        credentials: SardisCredentialsInput = CredentialsField(
            description="Sardis API credentials",
        )

    class Output(BlockSchemaOutput):
        balance: float = SchemaField(description="Current balance", default=0)
        remaining_limit: float = SchemaField(
            description="Remaining spending limit", default=0
        )
        token: str = SchemaField(description="Token type", default="USDC")
        error: str = SchemaField(description="Error message if failed", default="")

    def __init__(self):
        super().__init__(
            id="e9a2b3c4-5d6e-7f8a-9b0c-1d2e3f4a5b6c",
            description="Check the balance and remaining spending limits "
            "of a Sardis wallet.",
            categories={BlockCategory.OUTPUT},
            input_schema=SardisBalanceBlock.Input,
            output_schema=SardisBalanceBlock.Output,
            test_input=[
                {
                    "wallet_id": "wal_test123",
                    "token": "USDC",
                    "credentials": TEST_CREDENTIALS_INPUT,
                },
            ],
            test_output=[
                ("balance", 1000.0),
                ("remaining_limit", 500.0),
                ("token", "USDC"),
            ],
            test_mock={
                "get_balance": lambda *args, **kwargs: {
                    "balance": 1000.0,
                    "remaining_limit": 500.0,
                    "token": "USDC",
                }
            },
            test_credentials=TEST_CREDENTIALS,
        )

    @staticmethod
    async def get_balance(
        client: SardisClient, wallet_id: str, token: str
    ) -> dict:
        return await client.get_balance(wallet_id=wallet_id, token=token)

    async def run(
        self,
        input_data: Input,
        *,
        credentials: SardisCredentials,
        **kwargs,
    ) -> BlockOutput:
        client = SardisClient(credentials)
        result = await self.get_balance(
            client=client,
            wallet_id=input_data.wallet_id,
            token=input_data.token,
        )

        if "error" in result:
            yield "error", result["error"]
        else:
            yield "balance", float(result.get("balance", 0))
            yield "remaining_limit", float(result.get("remaining_limit", 0))
            yield "token", input_data.token
