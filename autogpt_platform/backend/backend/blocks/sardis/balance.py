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
    validate_wallet_id,
)
from backend.data.model import SchemaField


class SardisBalanceBlock(Block):
    """Check the balance and spending limits of a Sardis wallet."""

    class Input(BlockSchemaInput):
        wallet_id: str = SchemaField(
            description="Sardis wallet ID (starts with wal_)",
        )
        token: Literal["USDC", "USDT", "EURC", "PYUSD"] = SchemaField(
            description="Token to check",
            default="USDC",
        )
        credentials: SardisCredentialsInput = SardisCredentialsField()

        @field_validator("wallet_id")
        @classmethod
        def _validate_wallet_id(cls, v: str) -> str:
            return validate_wallet_id(v)

    class Output(BlockSchemaOutput):
        balance: str = SchemaField(
            description="Current balance (decimal string)", default="0"
        )
        remaining_limit: str = SchemaField(
            description="Remaining spending limit (decimal string)", default="0"
        )
        token: str = SchemaField(description="Token type", default="USDC")

    def __init__(self):
        super().__init__(
            id="ea396bee-d16f-42f6-9cb0-8ec7196351aa",
            description="Check the balance and remaining spending limits "
            "of a Sardis wallet.",
            categories={BlockCategory.DATA},
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
                ("balance", "1000.00"),
                ("remaining_limit", "500.00"),
                ("token", "USDC"),
            ],
            test_mock={
                "get_balance": lambda *args, **kwargs: {
                    "balance": "1000.00",
                    "remaining_limit": "500.00",
                    "token": "USDC",
                }
            },
            test_credentials=TEST_CREDENTIALS,
        )

    @staticmethod
    async def get_balance(client: SardisClient, wallet_id: str, token: str) -> dict:
        return await client.get_balance(wallet_id=wallet_id, token=token)

    async def run(
        self,
        input_data: Input,
        *,
        credentials: SardisCredentials,
        **kwargs,
    ) -> BlockOutput:
        client = await get_client(credentials)
        result = await self.get_balance(
            client=client,
            wallet_id=input_data.wallet_id,
            token=input_data.token,
        )

        if "error" in result:
            yield "error", str(result["error"])
        else:
            yield "balance", str(result.get("balance", "0"))
            yield "remaining_limit", str(result.get("remaining_limit", "0"))
            yield "token", input_data.token
