"""Spraay token balance check block.

Queries ``GET /api/v1/balances`` for a wallet's native ETH and ERC-20
balances on Base, the chain the gateway reads balances from.
"""

from typing import Any

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._api import spraay_request
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, spraay_provider


class SpraayGetBalanceBlock(Block):
    """Check a wallet's token balances on Base.

    Useful as a pre-flight check before preparing a batch payment, or for
    monitoring agent wallets. Costs $0.005 per call via x402.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = spraay_provider.credentials_field(
            description="Spraay x402 payer wallet private key",
        )
        wallet_address: str = SchemaField(
            description="Wallet address to check balances for",
        )
        token_address: str = SchemaField(
            description=(
                "ERC-20 contract address on Base to check. Leave empty to "
                "list ETH plus non-zero balances of popular Base tokens. "
                "USDC on Base: 0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
            ),
            default="",
        )

    class Output(BlockSchemaOutput):
        balances: list[dict[str, Any]] = SchemaField(
            description=(
                "All returned balances: symbol, address, decimals, "
                "balance (human decimal), balanceRaw (base units)"
            ),
        )
        balance: str = SchemaField(
            description="Requested token's balance in base units",
            default="",
        )
        formatted_balance: str = SchemaField(
            description="Requested token's balance with decimals applied",
            default="",
        )
        token_symbol: str = SchemaField(
            description="Requested token's symbol (e.g. USDC)",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="b81a036d-8bf7-460c-bca8-b1242d0aa3cb",
            description=(
                "Check native ETH and ERC-20 balances for any wallet on Base. "
                "Paid per call via Spraay x402."
            ),
            categories={BlockCategory.BASIC},
            effect=BlockEffect.READ,
            input_schema=SpraayGetBalanceBlock.Input,
            output_schema=SpraayGetBalanceBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "wallet_address": "0x1234567890abcdef1234567890abcdef12345678",
                "token_address": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            },
            test_output=[
                (
                    "balances",
                    [
                        {
                            "symbol": "ETH",
                            "address": "native",
                            "decimals": 18,
                            "balance": "0.01",
                            "balanceRaw": "10000000000000000",
                        },
                        {
                            "symbol": "USDC",
                            "address": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                            "decimals": 6,
                            "balance": "5.0",
                            "balanceRaw": "5000000",
                        },
                    ],
                ),
                ("balance", "5000000"),
                ("formatted_balance", "5.0"),
                ("token_symbol", "USDC"),
            ],
            test_mock={
                "fetch_balances": lambda *args, **kwargs: {
                    "address": "0x1234567890abcdef1234567890abcdef12345678",
                    "network": "base",
                    "chainId": 8453,
                    "balances": [
                        {
                            "symbol": "ETH",
                            "address": "native",
                            "decimals": 18,
                            "balance": "0.01",
                            "balanceRaw": "10000000000000000",
                        },
                        {
                            "symbol": "USDC",
                            "address": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                            "decimals": 6,
                            "balance": "5.0",
                            "balanceRaw": "5000000",
                        },
                    ],
                    "tokenCount": 2,
                }
            },
        )

    @staticmethod
    async def fetch_balances(
        private_key: str,
        wallet_address: str,
        token_address: str,
    ) -> dict[str, Any]:
        """Fetch wallet balances from the Spraay gateway."""
        params = {"address": wallet_address}
        if token_address:
            params |= {"tokens": token_address, "showAll": "true"}
        return await spraay_request(
            method="GET",
            endpoint="/api/v1/balances",
            private_key=private_key,
            params=params,
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        result = await self.fetch_balances(
            private_key=credentials.api_key.get_secret_value(),
            wallet_address=input_data.wallet_address,
            token_address=input_data.token_address,
        )

        balances = result.get("balances", [])
        yield "balances", balances

        if not input_data.token_address:
            return

        wanted = input_data.token_address.lower()
        match = next(
            (b for b in balances if b.get("address", "").lower() == wanted), None
        )
        if match is None or "error" in match:
            yield "error", (match or {}).get(
                "error", f"Token {input_data.token_address} not found in response"
            )
            return

        yield "balance", match.get("balanceRaw", "0")
        yield "formatted_balance", match.get("balance", "0")
        yield "token_symbol", match.get("symbol", "")
